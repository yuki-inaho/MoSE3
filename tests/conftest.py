"""Shared fixtures for the end-to-end tests."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def ckpt_dir() -> Path:
    return CKPT


@pytest.fixture
def teaser_frames(tmp_path: Path) -> list[Path]:
    """8 PNG frames extracted from assets/teaser_dress.gif, for the image-input path."""
    out_dir = tmp_path / "teaser_frames"
    out_dir.mkdir()
    frames = imageio.mimread(ROOT / "assets" / "teaser_dress.gif")
    step = max(1, len(frames) // 8)
    picked = frames[::step][:8]
    paths = []
    for i, frame in enumerate(picked):
        path = out_dir / f"{i:03d}.png"
        imageio.imwrite(path, np.asarray(frame)[..., :3])
        paths.append(path)
    return paths