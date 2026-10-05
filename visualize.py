"""Render 2D videos for one clip from ``inference.py`` (reads only its ``predictions.npz``).

    python visualize.py --clip outputs/spin --stride 48

A grid of query points (``--stride`` px apart, hiding points near depth edges) is drawn on the query
frame. Written next to ``predictions.npz``, each as ``input | visualization``:
  - ``query_points.png``  the grid on the query frame
  - ``tracks.mp4``        3D tracks with a fading trail, projected through each frame's camera
  - ``se3.mp4``           each point's SE(3) as a moving x/y/z frame (blue z = outward surface normal at
                          the query frame); frames of occluded points are dashed
  - ``rigidity.mp4``      rigidity embedding, PCA -> RGB
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

AXIS_COLORS = ((255, 32, 32), (0, 255, 64), (0, 180, 255))  # x, y, z


def grid_points(H: int, W: int, stride: int) -> np.ndarray:
    """[N, 2] integer (h, w) pixels on a centered grid."""
    hs = np.arange(stride // 2, H, stride)
    ws = np.arange(stride // 2, W, stride)
    hh, ww = np.meshgrid(hs, ws, indexing="ij")
    return np.stack([hh.ravel(), ww.ravel()], axis=1)


def point_colors(pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """[N, 3] uint8 RGB: hue from the column, brightness from the row."""
    rgb = [colorsys.hsv_to_rgb(w / max(W - 1, 1), 1.0, 0.8 + 0.2 * h / max(H - 1, 1)) for h, w in pts]
    return (np.array(rgb).reshape(-1, 3) * 255).round().astype(np.uint8)


def project(points_world: np.ndarray, camera_pose: np.ndarray, K: np.ndarray):
    """World points [..., 3] -> pixel coordinates uv [..., 2] and ok [...] (in front of the camera), through
    one camera-to-world pose [4, 4], or through each frame's own pose (points [T, ..., 3], poses [T, 4, 4])."""
    cam = camera_coords(points_world, camera_pose)
    z = cam[..., 2]
    ok = z > 1e-3
    z = np.where(ok, z, 1.0)
    uv = np.stack([K[0, 0] * cam[..., 0] / z + K[0, 2], K[1, 1] * cam[..., 1] / z + K[1, 2]], axis=-1)
    return uv, ok


def camera_coords(points_world: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """R^T (x - t): world points [..., 3] in the camera of ``camera_pose`` [4, 4], or of their frame
    (points [T, ..., 3], poses [T, 4, 4])."""
    pose = camera_pose.astype(np.float64)
    R, t = pose[..., :3, :3], pose[..., :3, 3]
    if pose.ndim == 3:                                                     # one pose per leading (frame) index
        extra = (1,) * (points_world.ndim - 2)
        R, t = R.reshape(len(R), *extra, 3, 3), t.reshape(len(t), *extra, 3)
    return ((points_world.astype(np.float64) - t)[..., None, :] @ R)[..., 0, :]


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """wxyz quaternions [..., 4] -> rotation matrices [..., 3, 3]."""
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def se3_apply(quat: np.ndarray, trans: np.ndarray, x_query: np.ndarray):
    """Move query-frame points by their SE(3): quat/trans [T, N, .], x_query [N, 3] -> (R x + t [T, N, 3], R)."""
    R = quat_to_R(quat.astype(np.float64))
    return np.einsum("tnij,nj->tni", R, x_query.astype(np.float64)) + trans, R


def surface_normal_frames(points_query: np.ndarray, cam_pos: np.ndarray, pts: np.ndarray, radius: int = 12,
                          edge: np.ndarray | None = None):
    """Per grid point, a frame [3, 3] (axes as columns) whose z is the outward surface normal at the query
    frame and whose x is the world x axis laid into the surface. Normals are averaged over a (2 radius + 1)^2
    window, on the point's own side of any depth ``edge``. Returns (frames [N, 3, 3], ok [N])."""
    P = cv2.GaussianBlur(points_query.astype(np.float32), (0, 0), 1.5).astype(np.float64)
    du, dv = np.zeros_like(P), np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(du, dv)
    n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12, None)
    n[(n * (cam_pos - P)).sum(-1) < 0] *= -1.0                             # face the camera
    valid = np.linalg.norm(n, axis=-1) > 0.5
    if edge is not None:
        valid &= ~_grow(edge, 4)                                           # the blur smears an edge over ~4 px

    frames, ok = np.tile(np.eye(3), (len(pts), 1, 1)), np.zeros(len(pts), bool)
    for i, (h, w) in enumerate(pts):
        h0, w0 = max(h - radius, 0), max(w - radius, 0)
        win = (slice(h0, h + radius + 1), slice(w0, w + radius + 1))
        sel = valid[win]
        if edge is not None:                                               # only the part connected to the point
            labels = cv2.connectedComponents(sel.astype(np.uint8), connectivity=4)[1]
            sel = sel & (labels == labels[h - h0, w - w0])
        if sel.sum() < 5:
            continue
        z = n[win][sel].mean(0)
        z /= max(np.linalg.norm(z), 1e-12)
        x = np.array([1.0, 0.0, 0.0]) - z * z[0]                           # world x, projected into the surface
        if np.linalg.norm(x) < 1e-8:                                       # normal along x: use the least aligned axis
            e = np.eye(3)[np.argmin(np.abs(z))]
            x = e - z * (e @ z)
        x /= np.linalg.norm(x)
        frames[i], ok[i] = np.stack([x, np.cross(z, x), z], axis=1), True
    return frames, ok


def depth_edges(z, rtol: float = 0.05) -> np.ndarray:
    """[H, W] bool: query pixels on a depth discontinuity (3x3 depth spread / depth above ``rtol``),
    i.e. object boundaries and the "flying" points between foreground and background."""
    pose = z["camera_poses"][int(z["query_idx"])].astype(np.float64)
    depth = ((z["tracks"][int(z["query_idx"])] - pose[:3, 3]) @ pose[:3, :3])[..., 2].astype(np.float32)  # z in the query camera
    kernel = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)
    return np.nan_to_num(spread / depth, nan=np.inf) > rtol


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), disk) > 0


def query_mask(z, edge_rtol: float = 0.05, edge_margin: int = 0) -> np.ndarray:
    """[H, W] bool: query pixels worth showing. Drops pixels within ``edge_margin`` px of a depth edge
    (see ``depth_edges``). ``edge_rtol`` = 0 switches the filter off."""
    keep = np.ones(z["tracks"].shape[1:3], bool)
    if edge_rtol > 0:
        keep &= ~_grow(depth_edges(z, edge_rtol), edge_margin)
    return keep


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def draw_points(image: np.ndarray, pts: np.ndarray, colors: np.ndarray, radius: int = 3) -> np.ndarray:
    out = image.copy()
    for (h, w), c in zip(pts, colors):
        cv2.circle(out, (int(w), int(h)), radius, c.tolist(), -1, cv2.LINE_AA)
    return out


def draw_trail(canvas, points, t, camera_pose, K, visible, colors, trail: int):
    """Draw on ``canvas`` (frame t, [H, W, 3], in place) the fading trail of world points [T, N, 3] over
    the last ``trail`` frames, projected through this frame's camera so camera motion does not show up
    in it. Segments while the point is occluded are drawn fainter. Returns (uv [N, 2], ok [N]) of frame t."""
    H, W = canvas.shape[:2]
    t0 = max(0, t - trail)
    uv, ok = project(points[t0:t + 1], camera_pose, K)                        # [k+1, N, 2] the trail window
    ok &= (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    for i, s in enumerate(range(t0, t)):  # oldest segment first; one blend per step, not per segment
        alpha = (i + 1) / (t - t0)
        seg_ok = ok[i] & ok[i + 1]
        seg_vis = visible[s] & visible[s + 1]
        for a, sel in ((alpha, seg_ok & seg_vis), (0.3 * alpha, seg_ok & ~seg_vis)):
            if not sel.any():
                continue
            layer = canvas.copy()
            for n in np.flatnonzero(sel):
                cv2.line(layer, _pt(uv[i, n]), _pt(uv[i + 1, n]), colors[n].tolist(), 2, cv2.LINE_AA)
            cv2.addWeighted(layer, a, canvas, 1 - a, 0, dst=canvas)
    return uv[-1], ok[-1]


def render_tracks(images, tracks, camera_poses, K, visibility, pts, colors, trail: int = 12) -> np.ndarray:
    """Tracks of the grid points drawn on the frames: a fading trail (see ``draw_trail``) and a head dot
    (hollow while the point is occluded). Returns [T, H, W, 3] uint8."""
    T = images.shape[0]
    hs, ws = pts[:, 0], pts[:, 1]
    visible = visibility[:, hs, ws] >= 0

    out = np.empty_like(images)
    for t in range(T):
        canvas = images[t].copy()
        uv, ok = draw_trail(canvas, tracks[:, hs, ws], t, camera_poses[t], K, visible, colors, trail)
        for n in np.flatnonzero(ok):
            cv2.circle(canvas, _pt(uv[n]), 3, colors[n].tolist(), -1 if visible[t, n] else 2, cv2.LINE_AA)
        out[t] = canvas
    return out


OCCLUDED_STYLES = ("alpha_dashed", "alpha", "lighten")


def _pale(rgb, amount=0.6):
    return tuple(int(round(c + (255 - c) * amount)) for c in rgb)


def _dashed_line(img, p0, p1, color, thickness, dash=7, gap=5):
    """cv2 has no dashed line: step along the segment drawing short pieces."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1:
        return
    u, start = (p1 - p0) / L, 0.0
    while start < L:
        end = min(start + dash, L)
        cv2.line(img, _pt(p0 + u * start), _pt(p0 + u * end), color, thickness, cv2.LINE_AA)
        start = end + gap


def render_se3_axes(images, tracks, se3_quat, se3_trans, se3_valid, visibility, camera_poses, K, query_idx, pts,
                    colors, axis_len: float = 20.0, thickness: int | None = None, normal_radius: int = 12,
                    dim: float = 0.8, occluded_style: str = "alpha_dashed", edge: np.ndarray | None = None,
                    trail: int = 0) -> np.ndarray:
    """Each grid point's SE(3) as a moving x/y/z frame at R x_query + t on the dimmed video, drawn in true
    perspective (``axis_len`` px for an axis in the image plane), occluded ones in ``occluded_style``;
    ``trail`` > 0 also draws the origin's trail. Returns [T, H, W, 3] uint8."""
    assert occluded_style in OCCLUDED_STYLES, occluded_style
    T, H, W, _ = images.shape
    thickness = max(3, round(0.1 * axis_len)) if thickness is None else thickness
    hs, ws = pts[:, 0], pts[:, 1]
    origin, R = se3_apply(se3_quat[:, hs, ws], se3_trans[:, hs, ws], tracks[query_idx, hs, ws])  # [T, N, 3], [T, N, 3, 3]
    keep = se3_valid[:, hs, ws].copy()
    if normal_radius > 0:
        start, ok = surface_normal_frames(tracks[query_idx], camera_poses[query_idx][:3, 3], pts, normal_radius, edge)
        R, keep = R @ start[None], keep & ok[None]
    occluded = visibility[:, hs, ws] < 0

    depth = camera_coords(origin, camera_poses)[..., 2]                    # z in each frame's camera
    length = axis_len * depth / K[0, 0]                                    # 3D length spanning axis_len px in-plane
    tips = origin[:, :, None, :] + length[..., None, None] * np.swapaxes(R, -1, -2)   # [T, N, 3 axes, 3]
    uv, ok = project(np.concatenate([origin[:, :, None, :], tips], 2), camera_poses, K)  # [T, N, 4, 2], [T, N, 4]
    tip_depth = camera_coords(tips, camera_poses)[..., 2]
    keep &= ok[:, :, 0] & (np.abs(uv[:, :, 0]) < 2 * max(H, W)).all(-1)    # in front of the camera, near the frame

    out = (images.astype(np.float32) * dim).astype(np.uint8)
    for t in range(T):
        frame = out[t]
        if trail > 0:
            draw_trail(frame, np.where(keep[..., None], origin, np.nan), t, camera_poses[t], K, ~occluded, colors, trail)
        for n in np.argsort(-depth[t]):                                    # far frames first
            if not keep[t, n]:
                continue
            o, occ = uv[t, n, 0], bool(occluded[t, n])
            pale = occ and occluded_style == "lighten"
            axis_colors = [_pale(c) for c in AXIS_COLORS] if pale else AXIS_COLORS
            rim = (150, 150, 150) if pale else (0, 0, 0)
            drawn = [k for k in np.argsort(-tip_depth[t, n])               # far axis first
                     if ok[t, n, 1 + k] and np.isfinite(uv[t, n, 1 + k]).all()
                     and np.linalg.norm(uv[t, n, 1 + k] - o) >= 2]

            target, shift, roi = frame, np.zeros(2), None
            if occ and occluded_style != "lighten":                        # real transparency: draw on a copy of the glyph's box
                box = np.array([o] + [uv[t, n, 1 + k] for k in drawn])
                x0, y0 = np.maximum(np.floor(box.min(0)).astype(int) - thickness - 4, 0)
                x1, y1 = np.minimum(np.ceil(box.max(0)).astype(int) + thickness + 5, [W, H])
                if x1 > x0 and y1 > y0:
                    roi = frame[y0:y1, x0:x1]
                    target, shift = roi.copy(), np.array([x0, y0], float)
            for k in drawn:
                a, b = o - shift, uv[t, n, 1 + k] - shift
                if occ and occluded_style == "alpha_dashed":
                    _dashed_line(target, a, b, rim, thickness + 1)
                    _dashed_line(target, a, b, axis_colors[k], max(1, thickness - 1))
                    cv2.arrowedLine(target, _pt(a + 0.7 * (b - a)), _pt(b), axis_colors[k], max(1, thickness - 1),
                                    cv2.LINE_AA, tipLength=0.8)
                else:
                    cv2.arrowedLine(target, _pt(a), _pt(b), rim, thickness + (2 if thickness >= 3 else 1),
                                    cv2.LINE_AA, tipLength=0.3)
                    cv2.arrowedLine(target, _pt(a), _pt(b), axis_colors[k], thickness, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(target, _pt(o - shift), thickness + 1, rim, -1, cv2.LINE_AA)
            cv2.circle(target, _pt(o - shift), max(1, thickness - 1), (255, 255, 255), -1, cv2.LINE_AA)
            if roi is not None:
                cv2.addWeighted(target, 0.5, roi, 0.5, 0, dst=roi)
    return out


def pca_rgb(rigidity_emb: np.ndarray, fit_points: int = 65536, seed: int = 0) -> np.ndarray:
    """Rigidity embedding [T, H, W, D] -> [T, H, W, 3] uint8: top-3 PCA of the unit-normalized
    embedding, each channel stretched between its 1st and 99th percentile."""
    T, H, W, D = rigidity_emb.shape

    def unit(x):
        x = np.nan_to_num(x.reshape(-1, D).astype(np.float32))
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-6, None)

    rng = np.random.default_rng(seed)
    flat = rigidity_emb.reshape(-1, D)
    sample = unit(flat[rng.choice(len(flat), size=min(fit_points, len(flat)), replace=False)])
    mean = sample.mean(0)
    _, vecs = np.linalg.eigh(np.cov((sample - mean).T))
    comps = vecs[:, ::-1][:, :3].T                                         # [3, D], largest first
    comps = comps * np.where(comps[np.arange(3), np.abs(comps).argmax(1)] < 0, -1.0, 1.0)[:, None]  # fixed sign

    lo, hi = np.percentile((sample - mean) @ comps.T, [1, 99], axis=0)
    out = np.empty((T, H, W, 3), np.uint8)
    for t in range(T):  # frame by frame keeps the peak memory low
        proj = (unit(rigidity_emb[t]) - mean) @ comps.T
        scaled = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255
        out[t] = scaled.reshape(H, W, 3).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Folder holding predictions.npz (an inference.py output folder).")
    p.add_argument("--stride", type=int, default=48, help="Grid spacing in pixels.")
    p.add_argument("--fps", type=float, help="Default: the fps stored in the clip's config.json.")
    p.add_argument("--trail", type=int, default=12, help="Trail length in frames (tracks.mp4).")
    p.add_argument("--axis_len", type=float, default=20.0,
                   help="SE(3) axis length in pixels, for an axis lying in the image plane.")
    p.add_argument("--normal_radius", type=int, default=12,
                   help="Lay each SE(3) frame on the surface at the query frame: z = outward normal, averaged over "
                        "this many pixels around the point. 0 = align the frames with the first camera instead.")
    p.add_argument("--dim", type=float, default=0.8, help="Brightness of the video under the SE(3) frames.")
    p.add_argument("--occluded_style", default="alpha_dashed", choices=OCCLUDED_STYLES,
                   help="SE(3) frames of occluded points: dashed + half transparent, half transparent, or pale colours.")
    p.add_argument("--edge_rtol", type=float, default=0.05,
                   help="Depth edge = relative 3x3 depth spread above this. Grid points near one are hidden, "
                        "and surface normals are taken from the point's own side of it. 0 = off.")
    p.add_argument("--edge_margin", type=int, default=4, help="Hide grid points within this many pixels of a depth edge.")
    args = p.parse_args()

    clip = Path(args.clip)
    z = np.load(clip / "predictions.npz")
    cfg_path = clip / "config.json"
    fps = args.fps or (json.loads(cfg_path.read_text()).get("fps", 8) if cfg_path.exists() else 8)

    images, tracks, poses, K, vis = z["images"], z["tracks"], z["camera_poses"], z["intrinsics"], z["visibility"]
    query_idx = int(z["query_idx"])
    T, H, W, _ = images.shape
    pts = grid_points(H, W, args.stride)
    edge = depth_edges(z, args.edge_rtol) if args.edge_rtol > 0 else None
    keep = query_mask(z, args.edge_rtol, args.edge_margin)[pts[:, 0], pts[:, 1]]
    print(f"[viz] {clip}: T={T} {W}x{H}, {keep.sum()} of {len(pts)} grid points shown (stride {args.stride})")
    pts = pts[keep]
    colors = point_colors(pts, H, W)

    def save(name, frames):
        imageio.mimsave(clip / name, list(np.concatenate([images, frames], axis=2)), fps=fps, macro_block_size=1)
        print(f"  wrote {clip / name}")

    imageio.imwrite(clip / "query_points.png", draw_points(images[query_idx], pts, colors))
    save("tracks.mp4", render_tracks(images, tracks, poses, K, vis, pts, colors, trail=args.trail))
    save("se3.mp4", render_se3_axes(images, tracks, z["se3_quat"], z["se3_trans"], z["se3_valid"], vis, poses, K,
                                    query_idx, pts, colors, axis_len=args.axis_len, normal_radius=args.normal_radius,
                                    dim=args.dim, occluded_style=args.occluded_style, edge=edge))
    save("rigidity.mp4", pca_rgb(z["rigidity_emb"]))


if __name__ == "__main__":
    main()
