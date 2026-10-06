"""End-to-end: the inference/visualization CLIs on the bundled examples and the teaser asset."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not (CKPT / "model.safetensors").exists(),
                       reason="model weights missing: hf download JoannaCCCCCC/MoSE3 --local-dir ckpt"),
]

EXPECTED_KEYS = {"images", "tracks", "visibility", "rigidity_emb", "se3_quat", "se3_trans",
                 "se3_valid", "camera_poses", "intrinsics", "query_idx"}
NUM_FRAMES = 8
MAX_SIDE = 266


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"cmd failed: {cmd}\n--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    return proc


def _check_predictions(out_dir: Path) -> tuple[int, int, int]:
    with np.load(out_dir / "predictions.npz") as z:
        assert set(z.files) == EXPECTED_KEYS
        images = z["images"]
        T, H, W, _ = images.shape
        assert T == NUM_FRAMES
        assert H % 14 == 0 and W % 14 == 0 and H <= MAX_SIDE and W <= MAX_SIDE
        assert images.dtype == np.uint8 and images.max() > 0
        assert z["tracks"].shape == (T, H, W, 3) and z["tracks"].dtype == np.float32
        assert z["rigidity_emb"].shape == (T, H, W, 16)
        assert z["se3_quat"].shape == (T, H, W, 4) and z["se3_quat"].dtype == np.float32
        assert z["se3_trans"].shape == (T, H, W, 3)
        assert z["visibility"].shape == (T, H, W)
        assert z["se3_valid"].shape == (T, H, W) and z["se3_valid"].dtype == bool
        assert z["camera_poses"].shape == (T, 4, 4)
        assert z["intrinsics"].shape == (3, 3)
        assert np.isfinite(z["tracks"]).all()
        assert np.isfinite(z["rigidity_emb"]).all()
        assert np.isfinite(z["se3_trans"]).all()
        assert z["intrinsics"][0, 0] > 0 and z["intrinsics"][1, 1] > 0
        assert np.allclose(z["camera_poses"][0], np.eye(4), atol=1e-4)

        q = int(z["query_idx"])
        assert 0 <= q < T
        valid = z["se3_valid"][q]
        assert valid.mean() > 0.5
        quat, trans = z["se3_quat"][q][valid], z["se3_trans"][q][valid]
        assert np.abs(quat[:, 1:]).max() < 5e-2
        assert np.linalg.norm(trans, axis=-1).max() < 5e-2
    return T, H, W


def test_inference_and_visualization_cli_end_to_end(tmp_path: Path, teaser_frames: list[Path]) -> None:
    frame_dir = teaser_frames[0].parent
    clips = [
        {"id": "spin", "video_path": str(ROOT / "examples" / "spin.mp4"), "num_frames": NUM_FRAMES,
         "max_w": MAX_SIDE, "max_h": MAX_SIDE, "se3_dist_weight": True,
         "output_dir": str(tmp_path / "spin")},
        {"id": "frames", "image_dir": str(frame_dir), "num_frames": NUM_FRAMES,
         "max_w": MAX_SIDE, "max_h": MAX_SIDE, "output_dir": str(tmp_path / "frames")},
    ]
    clips_json = tmp_path / "clips.json"
    clips_json.write_text(json.dumps(clips))

    _run([sys.executable, str(ROOT / "inference.py"), "--ckpt", str(CKPT), "--json", str(clips_json)], timeout=3600)

    shapes = {}
    for name in ("spin", "frames"):
        out = tmp_path / name
        assert (out / "predictions.npz").exists()
        assert (out / "video.mp4").stat().st_size > 0
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["id"] == name and len(cfg["frame_indices"]) == NUM_FRAMES
        shapes[name] = _check_predictions(out)

    _run([sys.executable, str(ROOT / "visualize.py"), "--clip", str(tmp_path / "spin"),
          "--stride", "64", "--normal_radius", "8"], timeout=1800)

    T, H, W = shapes["spin"]
    assert (tmp_path / "spin" / "query_points.png").stat().st_size > 0
    for name in ("tracks.mp4", "se3.mp4", "rigidity.mp4"):
        path = tmp_path / "spin" / name
        assert path.stat().st_size > 10_000, f"{name} too small"
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened(), f"{name} not readable"
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == T
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * W
        ok, frame = cap.read()
        cap.release()
        assert ok and frame.shape[:2] == (H, 2 * W)