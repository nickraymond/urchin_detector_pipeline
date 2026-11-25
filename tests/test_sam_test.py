"""Unit tests for the S1 SAM test helpers. Synthetic images only; no model, no real frames."""

import json

import numpy as np
import pytest
from PIL import Image

from app.sam_test import report, run
from app.sam_test.tiles import draw_overlay, tile_grid


def test_tile_grid_4k_covers_frame_with_overlap():
    tiles = tile_grid(3840, 2160)
    assert len(tiles) == 15  # 5 across x 3 down
    assert all(x1 - x0 == 1008 and y1 - y0 == 1008 for x0, y0, x1, y1 in tiles)
    assert max(t[2] for t in tiles) == 3840 and max(t[3] for t in tiles) == 2160
    xs = sorted({t[0] for t in tiles})
    assert all(1008 - (b - a) >= 0.2 * 1008 - 1 for a, b in zip(xs, xs[1:]))


def test_tile_grid_small_frame_is_one_tile():
    assert tile_grid(640, 480) == [(0, 0, 640, 480)]


def test_tile_grid_rejects_bad_overlap():
    with pytest.raises(ValueError):
        tile_grid(100, 100, overlap=1.0)


def test_draw_overlay_colours_mask_only():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    det = {"prompt": "fish", "score": 0.9, "box": [50, 40, 70, 60], "mask": mask}
    out = np.asarray(draw_overlay(img, [det], ["sea urchin", "fish"]))
    assert out[50, 60].sum() > 0  # inside the mask: coloured
    assert out[5, 5].sum() == 0  # far away: untouched


def _fake_result(tmp_path, status="ok"):
    Image.new("RGB", (64, 36), (0, 90, 120)).save(tmp_path / "o.jpg")
    frame = {"name": "frame1.png",
             "whole": {"seconds": 2.0, "tiles": 1, "masks": {"fish": 3}, "overlay": "o.jpg"},
             "tiles": {"seconds": 30.0, "tiles": 15, "masks": {"fish": 5}, "overlay": "o.jpg"}}
    return {"attempt": "mps", "device": "mps", "mps_fallback": False, "status": status,
            "error": "boom <bad>" if status != "ok" else None, "traceback": "Traceback...",
            "frames_done": [frame] if status == "ok" else [], "patches": ["decoder: x"],
            "peak_process_gb": 9.1, "peak_gpu_gb": 6.2}


def test_report_embeds_images_and_numbers(tmp_path):
    page = report.write_report(tmp_path, "clip.MP4", "sam3.1", ["fish"],
                               [_fake_result(tmp_path)])
    text = page.read_text()
    assert "data:image/jpeg;base64," in text
    assert "30.0" in text and "6.2" in text and "SAM ran on this laptop" in text
    assert "http://" not in text and "https://" not in text  # no internet assets


def test_report_shows_error_escaped(tmp_path):
    page = report.write_report(tmp_path, "clip.MP4", "sam3.1", ["fish"],
                               [_fake_result(tmp_path, status="error")])
    text = page.read_text()
    assert "did not complete" in text and "boom &lt;bad&gt;" in text


def test_run_attempt_survives_hard_crash(tmp_path, monkeypatch):
    class Crashed:
        returncode = -6
        stderr = "libc++abi: terminating\n"

    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: Crashed())
    r = run.run_attempt("mps", {"model": "sam3.1"}, tmp_path)
    assert r["status"] == "error" and "hard crash" in r["error"]
    assert "terminating" in r["stderr_tail"]


def test_run_attempt_reads_worker_result(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        job = json.loads(cmd[-1])
        assert kw["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        (tmp_path / "result_mps-fallback.json").write_text(json.dumps({"status": "ok"}))
        assert job["result_path"].endswith("result_mps-fallback.json")

        class Done:
            returncode, stderr = 0, ""
        return Done()

    monkeypatch.setattr(run.subprocess, "run", fake_run)
    assert run.run_attempt("mps-fallback", {}, tmp_path)["status"] == "ok"


def test_missing_config_gives_fix_hint(tmp_path):
    with pytest.raises(SystemExit, match="config.local.toml"):
        run.load_data_dir(tmp_path / "nope.toml")


def test_video_code_differs_for_same_name_in_two_folders(tmp_path):
    a, b = tmp_path / "x" / "dive 1.MP4", tmp_path / "y" / "dive 1.MP4"
    assert run.video_code(a) != run.video_code(b)
    assert run.video_code(a).startswith("dive_1_") and run.video_code(a) == run.video_code(a)


def test_index_lists_each_run_with_totals(tmp_path):
    run_dir = tmp_path / "clipA_abc123" / "sam3_tile504_cutoff0.5"
    run_dir.mkdir(parents=True)
    r = _fake_result(run_dir)
    (run_dir / "result_mps.json").write_text(json.dumps(r))
    report.write_report(run_dir, "clip.MP4", "sam3", ["fish"], [r], {"tile": 504, "cutoff": 0.5})
    assert "504 px tiles" in (run_dir / "report.html").read_text()
    text = report.write_index(tmp_path).read_text()
    assert "clipA_abc123/sam3_tile504_cutoff0.5/report.html" in text and "fish: 5" in text


def test_report_flags_missing_click_weights(tmp_path):
    r = _fake_result(tmp_path)
    r.update(click_weights_missing=40, click_weights_total=50,
             click_test={"error": "x", "traceback": ""})
    text = report.write_report(tmp_path, "c.MP4", "sam3.1", ["fish"], [r]).read_text()
    assert "missing 40 of 50" in text


def test_defaults_are_sam3_and_urchin():
    assert run.PROMPTS[0] == "urchin" and "sea urchin" not in run.PROMPTS
    assert run.MODELS["sam3"] == ("facebook/sam3", "sam3.pt")
