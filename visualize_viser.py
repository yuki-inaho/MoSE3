"""Interactive 3D viewer (viser) for one clip from ``inference.py`` (reads only its ``predictions.npz``).

    python visualize_viser.py --clip outputs/spin --stride 48 --port 8080

Shows the dense tracks as a moving point cloud (RGB or rigidity-embedding colours) and, on the same query
grid as ``visualize.py``, each point's SE(3) as an x/y/z tripod (optionally trailing its origin), plus
the trails of the predicted tracks. Sizes, lengths and trails are adjustable in the GUI.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import viser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize import (AXIS_COLORS, depth_edges, grid_points, pca_rgb, point_colors, query_mask, se3_apply,
                       surface_normal_frames)


def axis_segments(origin: np.ndarray, axis_dirs: np.ndarray, axis_len: float) -> np.ndarray:
    """Tripod line segments [T, N, 3 axes, 2 ends, 3] from origins [T, N, 3] and unit axis directions
    [T, N, 3 axes, 3]."""
    tips = origin[:, :, None, :] + axis_len * axis_dirs
    return np.stack([np.broadcast_to(origin[:, :, None, :], tips.shape), tips], axis=3).astype(np.float32)


def build_scene(z, stride: int = 48, point_stride: int = 2, axis_len: float = 0.05,
                edge_rtol: float = 0.05, edge_margin: int = 4, normal_radius: int = 12) -> dict:
    """Everything the viewer draws, as arrays in the first camera's frame. Query pixels on depth edges are
    left out (grid points also within ``edge_margin`` px); ``up`` is the direction tripods and trails are
    lifted off the surface."""
    tracks = z["tracks"]
    T, H, W, _ = tracks.shape
    q = int(z["query_idx"])
    lattice = (slice(0, H, point_stride), slice(0, W, point_stride))
    dense = query_mask(z, edge_rtol)[lattice].reshape(-1)
    grid = grid_points(H, W, stride)
    grid = grid[query_mask(z, edge_rtol, edge_margin)[grid[:, 0], grid[:, 1]]]
    hs, ws = grid[:, 0], grid[:, 1]

    origin, R = se3_apply(z["se3_quat"][:, hs, ws], z["se3_trans"][:, hs, ws], tracks[q, hs, ws])
    axes_ok = z["se3_valid"][:, hs, ws].copy()
    if normal_radius > 0:
        edge = depth_edges(z, edge_rtol) if edge_rtol > 0 else None
        start, ok = surface_normal_frames(tracks[q], z["camera_poses"][q][:3, 3], grid, normal_radius, edge)
        R, axes_ok = R @ start[None], axes_ok & ok[None]
    axis_dirs = np.swapaxes(R, -1, -2)                                     # [T, N, 3 axes, 3]: the columns of R
    # direction to lift the tripods / trails off the surface: the (moving) outward normal, or towards the camera
    up = axis_dirs[:, :, 2] if normal_radius > 0 else -origin / np.clip(np.linalg.norm(origin, axis=-1, keepdims=True), 1e-9, None)

    return {
        "points": tracks[(slice(None), *lattice)].reshape(T, -1, 3)[:, dense],
        "rgb": z["images"][q][lattice].reshape(-1, 3)[dense],
        "pca": pca_rgb(z["rigidity_emb"][q:q + 1])[0][lattice].reshape(-1, 3)[dense],
        "grid": grid,
        "grid_colors": point_colors(grid, H, W),
        "grid_tracks": tracks[:, hs, ws],                                  # [T, N, 3]
        "origin": origin.astype(np.float32),                               # [T, N, 3]  R x_query + t
        "axis_dirs": axis_dirs.astype(np.float32),                         # [T, N, 3, 3]
        "up": up.astype(np.float32),                                       # [T, N, 3]  unit lift direction
        "axes": axis_segments(origin, axis_dirs, axis_len),                # [T, N, 3, 2, 3]
        "axes_ok": axes_ok,                                                # [T, N]
    }


def trail_segments(points: np.ndarray, t: int, length: int) -> np.ndarray:
    """Line segments [N * k, 2, 3] joining the last ``k <= length`` steps of every point track [T, N, 3]
    up to frame t."""
    t0 = max(0, t - length)
    if t == t0:
        return np.zeros((0, 2, 3), np.float32)
    p = points[t0:t + 1]                                                   # [k+1, N, 3]
    return np.stack([p[:-1], p[1:]], axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 3).astype(np.float32)


def trail_colors(grid_colors: np.ndarray, n_segments: int) -> np.ndarray:
    """[n_segments, 2, 3] uint8: every point's colour repeated over its trail segments."""
    k = n_segments // max(len(grid_colors), 1)
    return np.repeat(grid_colors, k, axis=0)[:, None, :].repeat(2, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True, help="Folder holding predictions.npz (an inference.py output folder).")
    p.add_argument("--stride", type=int, default=48, help="Grid spacing in pixels (SE(3) tripods and trails).")
    p.add_argument("--point_stride", type=int, default=2, help="Keep every n-th pixel of the dense point cloud.")
    p.add_argument("--trail", type=int, default=12, help="Point-track trail length in frames.")
    p.add_argument("--se3_trail", type=int, default=0, help="Trail length of the SE(3) axes' origin in frames (0 = none).")
    p.add_argument("--trail_width", type=float, default=5.0, help="Trail line width in pixels (both trail layers).")
    p.add_argument("--axis_len", type=float, help="SE(3) axis length in world units (default: 3%% of the median depth).")
    p.add_argument("--axis_width", type=float, default=5.0, help="SE(3) axis line width in pixels.")
    p.add_argument("--lift", type=float,
                   help="Lift tripods, trails and track heads this far off the surface, along the outward normal "
                        "(default: 1%% of the median depth), so the point cloud does not bury them.")
    p.add_argument("--head_size", type=float, help="Size of the point-track heads (default: 3x the point size).")
    p.add_argument("--normal_radius", type=int, default=12,
                   help="Lay each SE(3) tripod on the surface at the query frame: z = outward normal, averaged over "
                        "this many pixels around the point. 0 = align the tripods with the first camera instead.")
    p.add_argument("--edge_rtol", type=float, default=0.05,
                   help="Hide query points on depth edges: relative 3x3 depth spread above this (0 = keep all).")
    p.add_argument("--edge_margin", type=int, default=4,
                   help="Also hide the grid points (tripods, trails) within this many pixels of a depth edge.")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    z = np.load(Path(args.clip) / "predictions.npz")
    depth = float(np.median(z["tracks"][int(z["query_idx"])][..., 2]))            # of the query-frame cloud
    axis_len = args.axis_len if args.axis_len is not None else 0.03 * depth
    lift = args.lift if args.lift is not None else 0.01 * depth
    point_size = 0.004 * depth
    head_size = args.head_size if args.head_size is not None else 3 * point_size
    s = build_scene(z, stride=args.stride, point_stride=args.point_stride, axis_len=axis_len,
                    edge_rtol=args.edge_rtol, edge_margin=args.edge_margin, normal_radius=args.normal_radius)
    T, N = s["grid_tracks"].shape[:2]
    print(f"[viser] {args.clip}: T={T}, {s['points'].shape[1]} points/frame, {N} grid points")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")  # coordinates are the first camera's: x right, y down, z forward
    H, W = z["tracks"].shape[1:3]
    fov = 2.0 * float(np.arctan(0.5 * H / float(z["intrinsics"][1, 1])))        # vertical, from the fitted K

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # start (and "Reset View") at the first camera, looking down +z at the cloud's median depth
        server.initial_camera.up = (0.0, -1.0, 0.0)
        server.initial_camera.fov = fov
        server.initial_camera.look_at = (0.0, 0.0, depth)
        server.initial_camera.position = (0.0, 0.0, 0.0)
        # a tab that reconnects keeps its old camera, so also set the live one (look_at last)
        client.camera.up_direction = (0.0, -1.0, 0.0)
        client.camera.fov = fov
        client.camera.position = (0.0, 0.0, 0.0)
        client.camera.look_at = (0.0, 0.0, depth)

    with server.gui.add_folder("Playback"):
        g_frame = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
        g_play = server.gui.add_checkbox("Play", initial_value=True)
        g_fps = server.gui.add_number("FPS", initial_value=8.0, min=0.5, max=60.0, step=0.5)
    with server.gui.add_folder("Point cloud"):
        g_color = server.gui.add_dropdown("Colour", ("RGB", "Rigidity embedding"), initial_value="RGB")
        g_psize = server.gui.add_slider("Point size", min=0.0005, max=0.05, step=0.0005, initial_value=point_size)
    with server.gui.add_folder("SE(3) tracks"):
        g_axes = server.gui.add_checkbox("Show axes", initial_value=True)
        g_alen = server.gui.add_slider("Axis length", min=0.005 * depth, max=0.3 * depth, step=0.005 * depth,
                                       initial_value=axis_len)
        g_awidth = server.gui.add_slider("Axis width", min=1.0, max=12.0, step=0.5, initial_value=args.axis_width)
        g_se3_tlen = server.gui.add_slider("Trail length", min=0, max=T - 1, step=1,
                                           initial_value=min(args.se3_trail, T - 1))
    with server.gui.add_folder("Point tracks"):
        g_trails = server.gui.add_checkbox("Show trails + heads", initial_value=False)
        g_hsize = server.gui.add_slider("Head size", min=0.0005, max=0.1, step=0.0005, initial_value=head_size)
        g_tlen = server.gui.add_slider("Trail length", min=0, max=T - 1, step=1, initial_value=min(args.trail, T - 1))
    with server.gui.add_folder("Trails (both layers)"):
        g_twidth = server.gui.add_slider("Trail width", min=1.0, max=16.0, step=0.5, initial_value=args.trail_width)
        g_lift = server.gui.add_slider("Lift off surface", min=0.0, max=0.1 * depth, step=0.0025 * depth,
                                       initial_value=lift)

    axis_rgb = np.broadcast_to(np.array(AXIS_COLORS, np.uint8)[None, :, None, :], (N, 3, 2, 3))
    layers = {"points": [], "axes": [], "se3_trails": [], "track_trails": [], "track_heads": []}

    def frame_arrays(t: int, origin, grid_tracks, axes, n_se3_trail: int, n_trail: int) -> dict:
        """(points, colors) of every grid layer at frame t, from the lifted origins / tracks."""
        ok = s["axes_ok"][t]
        se3, trk = trail_segments(origin, t, n_se3_trail), trail_segments(grid_tracks, t, n_trail)
        se3_ok = np.isfinite(se3).all(axis=(1, 2))
        return {"axes": (axes[t][ok].reshape(-1, 2, 3), axis_rgb[ok].reshape(-1, 2, 3)),
                "se3_trails": (se3[se3_ok], trail_colors(s["grid_colors"], len(se3))[se3_ok]),
                "track_trails": (trk, trail_colors(s["grid_colors"], len(trk))),
                "track_heads": (grid_tracks[t], s["grid_colors"])}

    def geometry():
        """Lifted SE(3) origins and tracks, tripod segments and the two trail lengths, from the sliders."""
        d = float(g_lift.value) * s["up"]
        origin = np.where(s["axes_ok"][..., None], s["origin"] + d, np.nan)  # hidden tripods leave no trail
        grid_tracks = s["grid_tracks"] + d
        return (origin, grid_tracks, axis_segments(origin, s["axis_dirs"], float(g_alen.value)),
                int(g_se3_tlen.value), int(g_tlen.value))

    def upload_points() -> None:
        for h in layers["points"]:
            h.remove()
        colors = s["rgb"] if g_color.value == "RGB" else s["pca"]
        layers["points"] = [
            server.scene.add_point_cloud(f"/points/{t:04d}", points=s["points"][t], colors=colors,
                                         point_size=float(g_psize.value), visible=False)
            for t in range(T)]

    upload_points()
    geo = geometry()
    for t in range(T):
        arrays = frame_arrays(t, *geo)
        for name, width in (("axes", args.axis_width), ("se3_trails", args.trail_width), ("track_trails", args.trail_width)):
            layers[name].append(server.scene.add_line_segments(
                f"/{name}/{t:04d}", points=arrays[name][0], colors=arrays[name][1], line_width=width, visible=False))
        layers["track_heads"].append(server.scene.add_point_cloud(
            f"/track_heads/{t:04d}", points=arrays["track_heads"][0], colors=arrays["track_heads"][1],
            point_size=head_size, visible=False))

    def refresh() -> None:
        geo = geometry()
        for t in range(T):
            arrays = frame_arrays(t, *geo)
            for name in ("axes", "se3_trails", "track_trails", "track_heads"):
                layers[name][t].points, layers[name][t].colors = arrays[name]

    def show(t: int) -> None:
        enabled = {"points": True, "axes": g_axes.value, "se3_trails": g_axes.value,
                   "track_trails": g_trails.value, "track_heads": g_trails.value}
        for name, handles in layers.items():
            for i, h in enumerate(handles):
                h.visible = (i == t) and enabled[name]

    g_frame.on_update(lambda _: show(int(g_frame.value)))
    g_axes.on_update(lambda _: show(int(g_frame.value)))
    g_trails.on_update(lambda _: show(int(g_frame.value)))
    for g in (g_alen, g_se3_tlen, g_tlen, g_lift):
        g.on_update(lambda _: refresh())

    @g_color.on_update
    def _(_) -> None:
        upload_points()
        show(int(g_frame.value))

    @g_psize.on_update
    def _(_) -> None:
        for h in layers["points"]:
            h.point_size = float(g_psize.value)

    @g_hsize.on_update
    def _(_) -> None:
        for h in layers["track_heads"]:
            h.point_size = float(g_hsize.value)

    @g_awidth.on_update
    def _(_) -> None:
        for h in layers["axes"]:
            h.line_width = float(g_awidth.value)

    @g_twidth.on_update
    def _(_) -> None:
        for h in layers["se3_trails"] + layers["track_trails"]:
            h.line_width = float(g_twidth.value)

    show(0)

    def play() -> None:
        while True:
            if g_play.value:
                g_frame.value = (int(g_frame.value) + 1) % T
            time.sleep(1.0 / max(float(g_fps.value), 0.5))

    threading.Thread(target=play, daemon=True).start()
    print(f"[viser] http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
