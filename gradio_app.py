"""Gradio app: drag & drop a video (or a set of frames) and get the MoSE3 visualizations back.

    python gradio_app.py --ckpt ckpt --port 7860

Outputs ``tracks.mp4`` (the main result: the predicted 3D tracks with fading trails), plus
``se3.mp4`` (the per-pixel SE(3) frames) and ``rigidity.mp4`` (the rigidity embedding), as on the
project page https://mose3-tracker.github.io/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import resolve_clip_config, run_clip, save_clip_outputs
from mose3.models.mose3 import MoSE3
from mose3.utils.video_io import load_image_dir_clip, load_video_clip
from visualize import render_clip

STATE = {"model": None, "device": None, "settings": {}}


def _video_fps(path: str, default: float = 8.0) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return min(max(fps, 1.0), 60.0) if fps and fps > 0 else default


def process(video_path, image_paths, num_frames, max_side, query_idx, dist_weight, stride):
    settings = STATE["settings"]
    if not video_path and not image_paths:
        raise gr.Error("Drop a video or a set of frames first.")
    work = Path(tempfile.mkdtemp(prefix="mose3_gradio_"))
    try:
        if video_path:
            source = {"video_path": str(video_path)}
            fps = _video_fps(video_path)
        else:
            frame_dir = work / "frames"
            frame_dir.mkdir()
            for i, p in enumerate(image_paths):
                shutil.copyfile(p, frame_dir / f"{i:04d}{Path(p).suffix.lower()}")
            source = {"image_dir": str(frame_dir)}
            fps = 8.0

        loader = load_video_clip if video_path else load_image_dir_clip
        load_kwargs = dict(num_frames=int(num_frames), max_w=int(max_side), max_h=int(max_side))
        frames_u8, _, _ = loader(source.get("video_path") or source["image_dir"], **load_kwargs)
        q = max(0, min(int(query_idx), int(frames_u8.shape[0]) - 1))

        cfg = resolve_clip_config({
            **source,
            "id": "upload",
            "output_dir": str(work / "clip"),
            "num_frames": int(num_frames),
            "max_w": int(max_side),
            "max_h": int(max_side),
            "query_idx": q,
            "se3_dist_weight": bool(dist_weight),
            "fps": fps,
        })
        payload = run_clip(STATE["model"], cfg, device=STATE["device"])
        out_dir = Path(cfg["output_dir"])
        save_clip_outputs(out_dir, cfg, payload)
        written = render_clip(out_dir, stride=int(stride))
        T, H, W, _ = payload["images"].shape
        status = (f"{T} frames at {W}x{H}, query frame {q}. "
                  f"Tracks -> tracks.mp4 (shown), SE(3) and rigidity under the accordions.")
        return str(written["tracks.mp4"]), str(written["se3.mp4"]), str(written["rigidity.mp4"]), status
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}")


def build_demo() -> gr.Blocks:
    settings = STATE["settings"]
    with gr.Blocks(title="MoSE3") as demo:
        gr.Markdown(
            "# MoSE3: Learning World-Space SE(3) at Every Pixel\n"
            "Drop a **monocular video** (or a set of frames) and get dense per-pixel 3D tracks, per-pixel "
            "SE(3) transforms and the rigidity embedding. "
            "[Project page](https://mose3-tracker.github.io/) · [Paper](https://arxiv.org/abs/2610.03716)"
        )
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Video", sources=["upload"], elem_id="input-video")
                frames = gr.File(label="or frames (ordered by file name)", file_count="multiple",
                                 file_types=["image"], type="filepath", elem_id="input-files")
                with gr.Row():
                    num_frames = gr.Slider(4, 64, value=settings["num_frames"], step=2, label="Frames sampled")
                    max_side = gr.Slider(154, 756, value=settings["max_side"], step=14, label="Max side (px)")
                with gr.Row():
                    query_idx = gr.Slider(0, 63, value=0, step=1, label="Query frame")
                    stride = gr.Slider(16, 96, value=settings["stride"], step=8, label="Grid stride (px)")
                dist_weight = gr.Checkbox(value=False, label="Distance-weighted SE(3) (locally deforming surfaces)")
                run_btn = gr.Button("Run", variant="primary", elem_id="run-btn")
                gr.Examples(examples=[[str(ROOT / "examples" / "spin.mp4")],
                                      [str(ROOT / "examples" / "butterfly.mp4")],
                                      [str(ROOT / "examples" / "folding_blanket.mp4")]],
                            inputs=[video])
            with gr.Column(scale=1):
                status = gr.Markdown("Drop an input and press **Run**.", elem_id="status")
                out_tracks = gr.Video(label="Tracks (main result)", elem_id="out-tracks")
                with gr.Accordion("SE(3) axes", open=False):
                    out_se3 = gr.Video(label="SE(3)", elem_id="out-se3")
                with gr.Accordion("Rigidity embedding", open=False):
                    out_rigidity = gr.Video(label="Rigidity", elem_id="out-rigidity")
        run_btn.click(process, [video, frames, num_frames, max_side, query_idx, dist_weight, stride],
                      [out_tracks, out_se3, out_rigidity, status])
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpt", help="Folder holding model.safetensors + config.json, or a Hub repo id.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--num_frames", type=int, default=32, help="Default frames sampled in the UI.")
    p.add_argument("--max_side", type=int, default=518, help="Default max side in the UI.")
    p.add_argument("--stride", type=int, default=48, help="Default grid stride in the UI.")
    args = p.parse_args()

    device = torch.device(args.device)
    STATE["device"] = device
    STATE["settings"] = {"num_frames": args.num_frames, "max_side": args.max_side, "stride": args.stride}
    print(f"[load] ckpt={args.ckpt} device={device}")
    STATE["model"] = MoSE3.from_pretrained(args.ckpt, strict=True).to(device).eval()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()