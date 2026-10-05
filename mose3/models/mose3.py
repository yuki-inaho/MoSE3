from copy import deepcopy

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .pi3 import Pi3
from .layers.block import TrackingBlockRope
from .layers.transformer_head import LinearPts3d, TemporalTrackingDecoder
from ..utils.geometry import fit_focal_from_bearings


class MoSE3(nn.Module, PyTorchModelHubMixin):
    """Dense 3D point tracks and rigidity embeddings from a tracking branch that runs alongside a frozen pi3.

    Per-pixel SE(3) is then fitted from them with ``mose3.utils.per_point_se3.fit_per_pixel_se3``.
    Load with ``MoSE3.from_pretrained(folder_or_hub_repo_id)``.
    """

    def __init__(self, track_start_layer: int = 10, rigidity_emb_dim: int = 16):
        super().__init__()
        # geometry branch (frozen)
        self.base = Pi3(pos_type="rope100", decoder_size="large")
        self.base.requires_grad_(False)
        self.track_start_layer = track_start_layer

        # tracking branch: one block per pi3 decoder layer from `track_start_layer` on
        self.tracking_blocks = nn.ModuleList([
            TrackingBlockRope(self.base.decoder[i])
            for i in range(track_start_layer, len(self.base.decoder))
        ])

        # point-tracking head (3D tracks + visibility), initialized from pi3's point-map head
        self.track_decoder = TemporalTrackingDecoder(deepcopy(self.base.point_decoder))
        self.track_head = deepcopy(self.base.point_head)
        self.vis_head = deepcopy(self.base.conf_head)

        # rigidity-embedding head
        self.rigid_decoder = deepcopy(self.base.point_decoder)
        self.rigid_head = LinearPts3d(
            patch_size=self.base.patch_size,
            dec_embed_dim=self.track_head.proj.in_features,
            output_dim=rigidity_emb_dim,
        )

        for head in (self.track_decoder, self.track_head, self.vis_head, self.rigid_decoder, self.rigid_head):
            head.requires_grad_(True)

    def forward(self, imgs, query_frame_idx=0):
        """
        Args:
            imgs: (B, N, 3, H, W) RGB in [0, 1]; H and W multiples of 14.
            query_frame_idx: frame whose pixels are tracked through the clip.

        Returns a dict. "Indexed by the query pixel" = entry (h, w) describes the scene point seen
        at pixel (h, w) of the query frame:
            tracks_local       (B, N, H, W, 3)  that point at frame n, in local camera coordinates
            visibility_logits  (B, N, H, W)     > 0 = visible at frame n
            rigidity_emb       (B, N, H, W, D)  rigidity embedding of frame n (indexed by frame-n pixels)
            tracks_pix_norm    (B, N, H, W, 2)  2D track, offset from the image centre / half-diagonal
            track_focal        (B, 2)           (fx, fy) in pixels, fitted to pi3's ray field
            camera_poses       (B, N, 4, 4)     camera-to-world, from pi3
            point_map          (B, N, H, W, 3)  pi3 point map of frame n (indexed by frame-n pixels)
            query_frame_idx    int
        """
        B, N, C, H, W = imgs.shape
        if not 0 <= query_frame_idx < N:
            raise ValueError(f"query_frame_idx must be in [0, {N}), got {query_frame_idx}")

        imgs = imgs.reshape(B * N, C, H, W)
        imgs = (imgs - self.base.image_mean) / self.base.image_std

        hidden = self.base.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        geom_feat, track_feat, pos, condition_idx = self._decode(hidden, N, H, W, query_frame_idx)

        camera_poses, point_map, conf, ray_xy = self._geometry_heads(geom_feat, pos, N, H, W)
        tracks_local, visibility_logits, pix_norm, track_focal = self._tracking_head(
            track_feat, pos, N, H, W, condition_idx, ray_xy, conf)
        rigidity_emb = self._rigidity_head(track_feat, pos, N, H, W)

        return {
            "tracks_local": tracks_local,
            "visibility_logits": visibility_logits,
            "rigidity_emb": rigidity_emb,
            "tracks_pix_norm": pix_norm,
            "track_focal": track_focal,
            "camera_poses": camera_poses,
            "point_map": point_map,
            "query_frame_idx": int(query_frame_idx),
        }

    def _decode(self, hidden, N, H, W, query_idx):
        """Run pi3's decoder (geometry branch) with the tracking branch alongside it."""
        B_N, _, D = hidden.shape
        B = B_N // N
        base = self.base

        # query-frame flag for the point-tracking head: 1 = query frame, 0 = any other frame
        condition_idx = torch.zeros(B * N, dtype=torch.long, device=hidden.device)
        condition_idx[torch.arange(B, device=hidden.device) * N + query_idx] = 1

        register_token = base.register_token.repeat(B, N, 1, 1).reshape(B * N, *base.register_token.shape[-2:])
        feat_geom = torch.cat([register_token, hidden], dim=1)

        # register tokens sit at position 0; patch positions are shifted by one
        pos = base.position_getter(B_N, H // base.patch_size, W // base.patch_size, hidden.device)
        pos_special = torch.zeros(B * N, base.patch_start_idx, 2).to(pos.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos + 1], dim=1)

        # even layers attend within a frame (B*N, seq, D), odd layers across frames (B, N*seq, D)
        for i in range(self.track_start_layer):
            is_local = (i % 2 == 0)
            p_cur = pos.reshape(B_N if is_local else B, -1, 2)
            f_in = feat_geom.reshape(B_N if is_local else B, -1, D)
            feat_geom = base.decoder[i](f_in, xpos=p_cur)

        feat_geom = feat_geom.reshape(B_N, -1, D)
        feat_track = feat_geom  # the tracking branch forks off here

        track_final, geom_final = [], []
        for i, track_blk in enumerate(self.tracking_blocks):
            layer_idx = self.track_start_layer + i
            is_local = (layer_idx % 2 == 0)
            p_cur = pos.reshape(B_N if is_local else B, -1, 2)
            f_in = feat_geom.reshape(B_N if is_local else B, -1, D)
            t_in = feat_track.reshape(B_N if is_local else B, -1, D)

            feat_geom = base.decoder[layer_idx](f_in, xpos=p_cur)
            feat_track = track_blk(x_track=t_in, x_geom=feat_geom, xpos_track=p_cur, xpos_geom=p_cur)

            # the heads read the last two layers, concatenated
            if layer_idx + 1 in [len(base.decoder) - 1, len(base.decoder)]:
                track_final.append(feat_track.reshape(B_N, -1, D))
                geom_final.append(feat_geom.reshape(B_N, -1, D))

        geom_final = torch.cat(geom_final, dim=-1)
        track_final = torch.cat(track_final, dim=-1)
        return geom_final, track_final, pos.reshape(B_N, -1, 2), condition_idx

    def _geometry_heads(self, feat_geom, pos, N, H, W):
        """pi3's own heads, unchanged: camera poses, per-frame point map, confidence."""
        B = feat_geom.shape[0] // N
        base = self.base

        camera_hidden = base.camera_decoder(feat_geom, xpos=pos)
        point_hidden = base.point_decoder(feat_geom, xpos=pos)
        conf_hidden = base.conf_decoder(feat_geom, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            camera_hidden = camera_hidden.float()
            point_hidden = point_hidden.float()

            camera_poses = base.camera_head(
                camera_hidden[:, base.patch_start_idx:], H // 14, W // 14
            ).reshape(B, N, 4, 4)

            res = base.point_head([point_hidden[:, base.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = res.split([2, 1], dim=-1)
            z = torch.exp(z)
            point_map = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = base.conf_head([conf_hidden[:, base.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

        return camera_poses, point_map, conf, xy

    def _tracking_head(self, feat_track, pos, N, H, W, condition_idx, ray_xy, conf):
        B = feat_track.shape[0] // N

        track_hidden = self.track_decoder(feat_track, xpos=pos, N=N, condition_idx=condition_idx)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            track_hidden = track_hidden.float()
            tokens = [track_hidden[:, self.base.patch_start_idx:]]

            # the head predicts ((u - cx) / half_diag, (v - cy) / half_diag, asinh(z))
            out = self.track_head(tokens, (H, W)).reshape(B, N, H, W, -1)
            pix_norm, z_raw = out.split([2, 1], dim=-1)
            focal = fit_focal_from_bearings(ray_xy, conf)  # (fx, fy) in pixels
            half_diag = 0.5 * (H ** 2 + W ** 2) ** 0.5
            z = torch.sinh(z_raw.clamp(-7.0, 7.0))
            tracks_local = torch.cat(
                [pix_norm * half_diag / focal.view(B, 1, 1, 1, 2) * z, z], dim=-1)

            visibility_logits = self.vis_head(tokens, (H, W)).reshape(B, N, H, W)

        return tracks_local, visibility_logits, pix_norm, focal

    def _rigidity_head(self, feat_track, pos, N, H, W):
        B = feat_track.shape[0] // N

        rigid_hidden = self.rigid_decoder(feat_track, xpos=pos)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            rigid_hidden = rigid_hidden.float()
            rigidity_emb = self.rigid_head(
                [rigid_hidden[:, self.base.patch_start_idx:]], (H, W)
            ).reshape(B, N, H, W, -1)

        return rigidity_emb
