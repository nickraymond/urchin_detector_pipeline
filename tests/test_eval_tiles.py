"""S8 Fixes 2 / D90: the eval factor and the tile plan (app/train/tiles.py), and frame_at. No torch."""

import numpy as np
import pytest
from PIL import Image

from app.train import samples, tiles


def sample(boxes, w=400, h=300, path=""):
    return {"image": path, "width": w, "height": h, "boxes": boxes, "ignore": []}


def test_eval_scale_is_44_over_the_median_shorter_side():
    got = tiles.eval_scale([sample([[0, 0, 0, 88, 200], [0, 0, 0, 40, 30]]), sample([[0, 0, 0, 100, 100]])])
    assert got == {"raw": 0.5, "factor": 0.5, "median_side_px": 88, "boxes": 3}   # sides 30, 88, 100
    even = tiles.eval_scale([sample([[0, 0, 0, 40, 40], [0, 0, 0, 48, 48]])])
    assert even["median_side_px"] == 44 and even["factor"] == 1.0


def test_eval_scale_is_clamped_to_trainings_bounds_and_none_without_boxes():
    tiny = tiles.eval_scale([sample([[0, 0, 0, 11, 11]])])
    assert tiny["raw"] == 4.0 and tiny["factor"] == 1.5
    huge = tiles.eval_scale([sample([[0, 0, 0, 4000, 4000]])])
    assert huge["factor"] == 0.03
    assert tiles.eval_scale([sample([])])["factor"] is None


@pytest.mark.parametrize("length, canvas, ov, want", [
    (96, 96, 20, [0]), (50, 96, 20, [0]), (200, 96, 20, [0, 76, 104]), (282, 96, 44, [0, 52, 104, 156, 186]),
])
def test_origins_cover_the_side_and_end_flush(length, canvas, ov, want):
    got = tiles.origins(length, canvas, ov)
    assert got == want and got[-1] + canvas >= length
    assert all(b - a <= canvas - ov for a, b in zip(got, got[1:]))     # every object up to `ov` lies whole in one


def test_every_box_up_to_the_overlap_lies_whole_in_some_tile():
    w, h, canvas, ov = 700, 450, 128, 40
    starts = tiles.plan(w, h, canvas, ov)
    for x in range(0, w - ov, 7):
        for y in range(0, h - ov, 11):
            assert any(ox <= x and x + ov <= ox + canvas and oy <= y and y + ov <= oy + canvas for ox, oy in starts)


def test_overlap_is_the_largest_box_at_the_factor_capped_at_half_a_canvas():
    train = [sample([[0, 0, 0, 50, 80]]), sample([[0, 0, 0, 30, 30]])]
    assert tiles.overlap(train, 0.5, 256) == 40
    assert tiles.overlap(train, 4.0, 256) == 128


def test_frame_at_resizes_the_whole_frame_and_its_boxes(tmp_path):
    Image.fromarray(np.zeros((300, 400, 3), np.uint8)).save(tmp_path / "f.png")
    s = {**sample([[3, 100, 50, 80, 40]], path=str(tmp_path / "f.png")), "ignore": [[0, 0, 20, 10]]}
    got = samples.frame_at(s, 0.5, 96)
    assert got["image"].shape == (150, 200, 3) and got["scale"] == 0.5
    assert got["boxes"] == [[3, 70.0, 35.0, 40.0, 20.0]] and got["ignore"] == [[5.0, 2.5, 10.0, 5.0]]
    old = samples.frame_at(s, None, 96)                        # no factor: shrunk to fit, as before
    assert old["image"].shape == (72, 96, 3)
