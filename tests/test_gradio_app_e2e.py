"""End-to-end: the Gradio app driven in a real browser with Playwright (video and frames inputs)."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ckpt"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not (CKPT / "model.safetensors").exists(),
                       reason="model weights missing: hf download JoannaCCCCCC/MoSE3 --local-dir ckpt"),
]

playwright_api = pytest.importorskip("playwright.sync_api")

TIMEOUT_MS = 900_000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def app_url(tmp_path_factory):
    port = _free_port()
    log_path = tmp_path_factory.mktemp("gradio") / "app.log"
    log = log_path.open("w")
    env = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "gradio_app.py"), "--ckpt", str(CKPT), "--port", str(port),
         "--num_frames", "8", "--max_side", "266", "--stride", "64"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 900
    ready = False
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    if not ready:
        log.flush()
        proc.terminate()
        raise RuntimeError(f"gradio app not ready (exit={proc.poll()}):\n{log_path.read_text()[-4000:]}")
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        yield browser
        browser.close()


def _new_page(browser, app_url: str):
    page = browser.new_page()
    page.set_default_timeout(TIMEOUT_MS)
    page.goto(app_url, wait_until="domcontentloaded")
    page.wait_for_selector("#run-btn")
    return page


def _drop_and_run(page, file_selector: str, files) -> bytes:
    page.set_input_files(file_selector, files)
    page.click("#run-btn")
    page.wait_for_function(
        "() => { const v = document.querySelector('#out-tracks video'); return !!v && !!v.src; }")
    assert "query frame" in page.inner_text("#status")
    src = page.eval_on_selector("#out-tracks video", "v => v.src")
    response = page.request.get(src)
    assert response.ok, response.status
    body = response.body()
    assert len(body) > 20_000, f"tracks video too small: {len(body)} bytes"
    assert b"ftyp" in body[:64], "not an mp4"
    return body


def test_drag_and_drop_video(browser, app_url: str) -> None:
    page = _new_page(browser, app_url)
    assert "MoSE3" in page.title()
    _drop_and_run(page, "#input-video input[type=file]", str(ROOT / "examples" / "spin.mp4"))
    page.close()


def test_drag_and_drop_frames(browser, app_url: str, teaser_frames: list[Path]) -> None:
    page = _new_page(browser, app_url)
    _drop_and_run(page, "#input-files input[type=file]", [str(p) for p in teaser_frames[:4]])
    page.close()