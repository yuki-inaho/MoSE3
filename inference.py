"""Run MoSE3 on a video (or image folder) and save the predictions.

``--ckpt`` is a folder holding ``model.safetensors`` + ``config.json``, or a Hugging Face repo id.

One clip from the command line, from a video or from a folder of frames (ordered by file name number):

    python inference.py --ckpt ckpt --video examples/spin.mp4
    python inference.py --ckpt ckpt --image_dir path/to/frames

or several clips from a JSON list (same keys as the flags; see ``DEFAULTS``):

    python inference.py --ckpt ckpt --json examples/demo.json

Each clip writes to ``output_dir`` (default ``outputs/<id>`` next to this file):
  - ``predictions.npz``   the predictions (see ``run_clip`` for the keys)
  - ``config.json``       resolved clip config, incl. the sampled source frame indices
  - ``video.mp4``         the sampled input frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch

from mose3.models.mose3 import MoSE3
from mose3.utils.video_io import load_image_dir_clip, load_video_clip
from mose3.utils.per_point_se3 import fit_per_pixel_se3

ROOT = Path(__file__).resolve().parent

DEFAULTS: Dict[str, Any] = {
    "num_frames": 50,
    "start_frame": 0,
    "frame_interval": 0,        # 0 = spread num_frames uniformly over the clip, else a fixed stride
    "query_idx": 0,             # query frame: its pixels are the ones tracked
    "max_w": 518,
    "max_h": 518,
    "central_crop": False,      # crop to a centered square before resizing
    "num_ref_pts": 12000,       # SE(3) fit: size of the shared reference-point pool
    "se3_temp": 0.01,           # SE(3) fit: softmax temperature on rigidity-embedding similarity
    "se3_dist_weight": False,   # SE(3) fit: also weight the pool by query-frame pixel distance
    "se3_sigma_px": 8.0,        # SE(3) fit: std of that distance weight, in pixels
    "fps": 8,
    "id": None,                 # default: file / folder name
    "output_dir": None,         # default: outputs/<id>
}


def resolve_clip_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    # fill DEFAULTS for every key the clip did not set, then derive id / output_dir
    source_keys = [k for k in ("video_path", "image_dir") if raw.get(k) is not None]
    if len(source_keys) != 1:
        raise ValueError(f"clip config must include exactly one of video_path/image_dir: {raw}")
    cfg = {**DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}
    if cfg["id"] is None:
        cfg["id"] = Path(cfg[source_keys[0]]).stem
    if cfg["output_dir"] is None:
        cfg["output_dir"] = str(ROOT / "outputs" / cfg["id"])
    return cfg


def _to_cpu_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_first_camera_frame(points_local: torch.Tensor, camera_poses: torch.Tensor):
    """pi3 predicts camera-to-world poses [T,4,4] in its own world frame. Re-express them in the first
    camera's frame (frame 0 becomes the identity) and lift per-frame camera points [T,H,W,3] into it."""
    camera_poses = torch.linalg.inv(camera_poses[0:1]) @ camera_poses
    R, t = camera_poses[:, :3, :3], camera_poses[:, :3, 3]
    points = torch.einsum("tij,thwj->thwi", R, points_local) + t[:, None, None, :]
    return points, camera_poses


@torch.inference_mode()
def run_clip(model, clip_cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Returns the ``predictions.npz``: T frames at H x W. All 3D quantities are expressed in the
    coordinate frame of the first camera (frame 0), which is what "world" means below.
    Dense arrays are indexed by the query-frame pixel (h, w); ``rigidity_emb[t]`` by frame-t pixels.

        images        [T,H,W,3]   uint8
        tracks        [T,H,W,3]   world-frame 3D position of every query pixel's point, at every frame
        visibility    [T,H,W]     logit, > 0 = visible
        rigidity_emb  [T,H,W,D]   rigidity embedding
        se3_quat      [T,H,W,4]   per-pixel rigid motion query frame -> t in world coordinates:
        se3_trans     [T,H,W,3]     x_t = R(se3_quat) x_query + se3_trans      (quaternion is wxyz)
        se3_valid     [T,H,W]     the closed-form fit was well conditioned
        camera_poses  [T,4,4]     camera-to-world
        intrinsics    [3,3]       pinhole K (pixels): the model's fitted focal, principal point at the image centre
        query_idx     int         index of the query frame among the T frames; its pixels are the ones tracked
    """
    # 1. load the sampled frames from the video or the image folder
    loader, source = ((load_image_dir_clip, clip_cfg["image_dir"]) if clip_cfg.get("image_dir") is not None
                      else (load_video_clip, clip_cfg["video_path"]))
    kwargs = dict(
        num_frames=clip_cfg["num_frames"],
        start_frame=clip_cfg["start_frame"],
        frame_interval=clip_cfg["frame_interval"],
        max_w=clip_cfg["max_w"],
        max_h=clip_cfg["max_h"],
        central_crop=bool(clip_cfg["central_crop"]),
    )
    if loader is load_video_clip:  # a JSON clip may pin exact source frames
        kwargs["frame_indices"] = clip_cfg.get("frame_indices")
    frames_u8, frame_indices, _ = loader(source, **kwargs)
    clip_cfg["frame_indices"] = [int(i) for i in frame_indices]  # recorded in config.json
    T, H, W, _ = frames_u8.shape
    images_b = (  # [T,H,W,3] uint8 -> [1,T,3,H,W] float in [0,1]
        torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float() / 255.0
    ).unsqueeze(0)

    # 2. forward pass: 3D tracks of the query pixels, rigidity embeddings, camera poses
    q = int(clip_cfg["query_idx"])
    print(f"  forward: T={T} H={H} W={W} query={q}")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(images_b, query_frame_idx=q)

    tracks_local = out["tracks_local"][0].float()
    rigidity_emb = out["rigidity_emb"][0].float()
    tracks, camera_poses = _to_first_camera_frame(tracks_local, out["camera_poses"][0].float())

    # pinhole K from the focal the model fitted to decode its tracks (pixel w sits at offset w + 0.5 - W/2)
    fx, fy = out["track_focal"][0].float().tolist()
    intrinsics = np.array([[fx, 0, 0.5 * W - 0.5], [0, fy, 0.5 * H - 0.5], [0, 0, 1]], np.float32)

    # 3. per-pixel SE(3): closed-form rigid fit over the tracks, weighted by rigidity-embedding similarity
    sigma_px = float(clip_cfg["se3_sigma_px"]) if clip_cfg["se3_dist_weight"] else 0.0
    print(f"  per-pixel SE(3): N_ref={clip_cfg['num_ref_pts']} temp={clip_cfg['se3_temp']} sigma_px={sigma_px:g}")
    se3 = fit_per_pixel_se3(
        tracks=tracks,
        rigidity_emb=rigidity_emb,
        query_idx=q,
        num_ref_pts=int(clip_cfg["num_ref_pts"]),
        temperature=float(clip_cfg["se3_temp"]),
        sigma_px=sigma_px,
        device=device,
    )

    return {
        "images": frames_u8,
        "tracks": _to_cpu_np(tracks),
        "visibility": _to_cpu_np(out["visibility_logits"][0].float()),
        "rigidity_emb": _to_cpu_np(rigidity_emb),
        "se3_quat": _to_cpu_np(se3["quat"]),
        "se3_trans": _to_cpu_np(se3["trans"]),
        "se3_valid": _to_cpu_np(se3["ok"]),
        "camera_poses": _to_cpu_np(camera_poses),
        "intrinsics": intrinsics,
        "query_idx": np.int64(q),
    }


def save_clip_outputs(out_dir: Path, cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    # predictions.npz + the sampled input frames as video.mp4 + the resolved clip config
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "predictions.npz", **payload)
    imageio.mimwrite(out_dir / "video.mp4", payload["images"], fps=cfg["fps"], quality=8, macro_block_size=1)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Folder holding model.safetensors + config.json, or a Hugging Face repo id.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", dest="video_path", help="Input video.")
    src.add_argument("--image_dir", help="Folder of frames, ordered by the numbers in their file names.")
    src.add_argument("--json", help="JSON list of clip configs, to run several clips.")
    p.add_argument("--output_dir", help="Default: outputs/<video or folder name>.")
    p.add_argument("--num_frames", type=int, help=f"Default {DEFAULTS['num_frames']}.")
    p.add_argument("--start_frame", type=int)
    p.add_argument("--frame_interval", type=int,
                   help="Frame stride; 0 (default) spreads num_frames uniformly over the clip.")
    p.add_argument("--query_idx", type=int, help="Query frame among the sampled frames (default 0).")
    p.add_argument("--max_w", type=int, help=f"Default {DEFAULTS['max_w']}.")
    p.add_argument("--max_h", type=int, help=f"Default {DEFAULTS['max_h']}.")
    p.add_argument("--central_crop", action="store_true", default=None,
                   help="Crop frames to a centered square before resizing.")
    p.add_argument("--num_ref_pts", type=int,
                   help=f"SE(3) fit: query-frame pixels sampled as the reference pool (default {DEFAULTS['num_ref_pts']}).")
    p.add_argument("--se3_dist_weight", action=argparse.BooleanOptionalAction, default=None,
                   help="SE(3) fit: also weight the pool by pixel distance, for locally deforming surfaces (default off).")
    p.add_argument("--se3_sigma_px", type=float,
                   help=f"SE(3) fit: sigma of that distance weight, in pixels (default {DEFAULTS['se3_sigma_px']:g}).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # several clips from a JSON list, or a single clip from the flags
    if args.json is not None:
        raw_clips = json.loads(Path(args.json).read_text())
        if not isinstance(raw_clips, list):
            raise ValueError("JSON must be a list of clip dicts")
    else:
        raw_clips = [{k: getattr(args, k) for k in (
            "video_path", "image_dir", "output_dir", "num_frames", "start_frame", "frame_interval",
            "query_idx", "max_w", "max_h", "central_crop", "num_ref_pts", "se3_dist_weight", "se3_sigma_px")}]
    clips = [resolve_clip_config(c) for c in raw_clips]

    device = torch.device(args.device)
    print(f"[load] ckpt={args.ckpt} device={device}")
    model = MoSE3.from_pretrained(args.ckpt, strict=True).to(device).eval()  # local folder or Hub repo id

    for i, cfg in enumerate(clips):
        print(f"[{i+1}/{len(clips)}] id={cfg['id']} → {cfg['output_dir']}")
        payload = run_clip(model, cfg, device=device)
        save_clip_outputs(Path(cfg["output_dir"]), cfg, payload)
        print(f"  wrote {Path(cfg['output_dir']) / 'predictions.npz'}")


if __name__ == "__main__":
    main()
