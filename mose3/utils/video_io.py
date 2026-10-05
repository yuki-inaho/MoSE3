"""Video/image-folder → uint8 tensor with max_w/max_h cap and round-to-14."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _round14(x: int) -> int:
    return max(14, int(round(x / 14.0)) * 14)


def _target_size(orig_w: int, orig_h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    """Cap (orig_w, orig_h) by (max_w, max_h) preserving aspect, then round each to 14."""
    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
    tw = max(14, _round14(int(round(orig_w * scale))))
    th = max(14, _round14(int(round(orig_h * scale))))
    while tw > max_w:
        tw -= 14
    while th > max_h:
        th -= 14
    return tw, th


def _central_square_crop_bounds(width: int, height: int) -> Tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` for a centered square crop."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    side = min(width, height)
    x0 = max(0, (width - side) // 2)
    y0 = max(0, (height - side) // 2)
    return x0, y0, x0 + side, y0 + side


def select_frame_indices(
    total_frames: int, num_frames: int, start_frame: int, frame_interval: int,
) -> List[int]:
    """Pick `num_frames` indices.

    `frame_interval == 0` ⇒ uniformly spread `num_frames` indices across
    [start_frame, total_frames-1]. Otherwise step by `frame_interval` from
    `start_frame`. Always clamps to [0, total_frames-1] and dedups while
    preserving order.
    """
    if total_frames <= 0:
        raise ValueError("video has no frames")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    start = max(0, min(start_frame, total_frames - 1))
    if frame_interval == 0:
        if num_frames == 1:
            return [start]
        last = total_frames - 1
        if last <= start:
            return [start] * num_frames
        idxs = np.linspace(start, last, num=num_frames).round().astype(int).tolist()
    else:
        idxs = [start + i * frame_interval for i in range(num_frames)]
        idxs = [min(max(0, i), total_frames - 1) for i in idxs]
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(int(i))
    return out


def load_video_clip(
    video_path: str | Path,
    num_frames: int,
    start_frame: int = 0,
    frame_interval: int = 0,
    max_w: int = 518,
    max_h: int = 518,
    central_crop: bool = False,
    frame_indices: List[int] | None = None,
) -> Tuple[np.ndarray, List[int], Tuple[int, int]]:
    """Returns (frames_uint8 [T, H, W, 3], picked_indices, (orig_w, orig_h))."""
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    picked = (select_frame_indices(total, num_frames, start_frame, frame_interval)
              if frame_indices is None else list(frame_indices))
    if (not picked or (frame_indices is not None and len(picked) != num_frames)
            or any(not isinstance(i, (int, np.integer)) for i in picked)
            or picked[0] < 0 or picked[-1] >= total
            or any(b <= a for a, b in zip(picked, picked[1:]))):
        cap.release()
        raise ValueError('Explicit video frame indices must be ordered, unique, and within the video')
    picked_set = set(picked)

    raw = {}
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx in picked_set:
            raw[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        idx += 1
        if idx > max(picked):
            break
    cap.release()
    if len(raw) != len(picked):
        raise RuntimeError(
            f"video {path}: expected {len(picked)} frames, decoded {len(raw)}"
        )

    crop_bounds = _central_square_crop_bounds(orig_w, orig_h) if central_crop else None
    resize_w = crop_bounds[2] - crop_bounds[0] if crop_bounds is not None else orig_w
    resize_h = crop_bounds[3] - crop_bounds[1] if crop_bounds is not None else orig_h
    tw, th = _target_size(resize_w, resize_h, max_w, max_h)
    frames = []
    for i in picked:
        f = raw[i]
        if crop_bounds is not None:
            x0, y0, x1, y1 = crop_bounds
            f = f[y0:y1, x0:x1]
        if (tw, th) != (f.shape[1], f.shape[0]):
            f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        frames.append(f)
    return np.stack(frames, axis=0).astype(np.uint8), picked, (orig_w, orig_h)


def _image_files(image_dir: str | Path) -> List[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {root}")
    files = sorted(  # by the numbers in the name: 2.png < 10.png, frame_2.png < frame_10.png
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)],
    )
    if not files:
        raise ValueError(f"image_dir has no supported image files: {root}")
    return files


def load_image_dir_clip(
    image_dir: str | Path,
    num_frames: int,
    start_frame: int = 0,
    frame_interval: int = 0,
    max_w: int = 518,
    max_h: int = 518,
    central_crop: bool = False,
) -> Tuple[np.ndarray, List[int], Tuple[int, int]]:
    """Returns (frames_uint8 [T, H, W, 3], picked_indices, (orig_w, orig_h))."""
    files = _image_files(image_dir)
    picked = select_frame_indices(len(files), num_frames, start_frame, frame_interval)

    first = cv2.imread(str(files[picked[0]]), cv2.IMREAD_COLOR)
    if first is None:
        raise IOError(f"cannot read image: {files[picked[0]]}")
    orig_h, orig_w = first.shape[:2]
    crop_bounds = _central_square_crop_bounds(orig_w, orig_h) if central_crop else None
    resize_w = crop_bounds[2] - crop_bounds[0] if crop_bounds is not None else orig_w
    resize_h = crop_bounds[3] - crop_bounds[1] if crop_bounds is not None else orig_h
    tw, th = _target_size(resize_w, resize_h, max_w, max_h)

    frames = []
    for i in picked:
        bgr = first if i == picked[0] else cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read image: {files[i]}")
        if bgr.shape[:2] != (orig_h, orig_w):
            raise ValueError(
                f"image {files[i]} has size {bgr.shape[1]}x{bgr.shape[0]}, "
                f"expected {orig_w}x{orig_h}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if crop_bounds is not None:
            x0, y0, x1, y1 = crop_bounds
            rgb = rgb[y0:y1, x0:x1]
        if (tw, th) != (rgb.shape[1], rgb.shape[0]):
            rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    return np.stack(frames, axis=0).astype(np.uint8), picked, (orig_w, orig_h)
