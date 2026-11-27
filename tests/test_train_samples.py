"""S8 bite 4b (first part): training samples and augmentation (app/train/samples.py), on synthetic
images. Test ideas reused from the earlier work's test_data_aug.py: labels do not move when blur
is turned on, the same seed gives the same sample, evaluation is never augmented."""

import numpy as np
import pytest
from PIL import Image

from app.train import samples


@pytest.fixture
def sample(tmp_path):
    """A 1200 x 800 frame: flat blue water with three bright 100 px squares ("urchins") and
    one ignored region."""
    a = np.zeros((800, 1200, 3), np.uint8)
    a[:] = (20, 60, 120)
    boxes = [[1, 200, 150, 100, 100], [1, 700, 400, 100, 100], [3, 1000, 600, 100, 100]]
    for _, x, y, w, h in boxes:
        a[y:y + h, x:x + w] = (230, 200, 60)
    p = tmp_path / "frame.png"
    Image.fromarray(a).save(p)
    return {"image": str(p), "width": 1200, "height": 800, "boxes": boxes,
            "ignore": [[400, 100, 80, 80]]}


def test_same_seed_epoch_and_index_give_the_same_sample(sample):
    a = samples.make(sample, 256, seed=3, epoch=2, index=5)
    b = samples.make(sample, 256, seed=3, epoch=2, index=5)
    assert np.array_equal(a["image"], b["image"]) and a["boxes"] == b["boxes"]
    c = samples.make(sample, 256, seed=3, epoch=3, index=5)
    assert not np.array_equal(a["image"], c["image"])          # another epoch: another view
    assert a["image"].shape == (256, 256, 3) and a["image"].dtype == np.uint8


def test_boxes_stay_on_their_objects_inside_the_canvas(sample):
    """Over many views: every box is inside the canvas, at least 8 px a side, and the pixels
    under its centre are the bright square (geometry and labels move together, flips too)."""
    seen = 0
    for i in range(40):
        got = samples.make(sample, 256, seed=1, epoch=0, index=i, preset="standard")
        assert 0.03 <= got["scale"] <= 1.5
        for cls, cx, cy, w, h in got["boxes"]:
            assert cls in (1, 3) and w >= samples.MIN_BOX and h >= samples.MIN_BOX
            assert 0 <= cx - w / 2 and cx + w / 2 <= 256 and 0 <= cy - h / 2 and cy + h / 2 <= 256
            r, g, b = got["image"][int(cy), int(cx)].astype(int)
            assert r > b, (i, got["boxes"])                    # the yellow square, not blue water
            seen += 1
        # the median target lands in the band (24-64 px) unless the canvas edge cut it
        whole = [min(w, h) for _, cx, cy, w, h in got["boxes"]
                 if 2 < cx - w / 2 and cx + w / 2 < 254 and 2 < cy - h / 2 and cy + h / 2 < 254]
        assert all(samples.BAND[0] - 2 <= s <= samples.BAND[1] + 2 for s in whole)
    assert seen > 40


def test_blur_and_haze_never_move_a_box(sample):
    """Their random streams are their own: the same seed gives the same boxes with or without."""
    changed = 0
    for i in range(30):
        plain = samples.make(sample, 256, seed=9, epoch=1, index=i, preset="standard")
        both = samples.make(sample, 256, seed=9, epoch=1, index=i, preset="standard + blur + haze")
        assert plain["boxes"] == both["boxes"] and plain["ignore"] == both["ignore"]
        changed += not np.array_equal(plain["image"], both["image"])
    assert 5 <= changed < 30        # each applies to about 30 % of the samples, not to all


def test_haze_lowers_contrast_and_blur_softens_edges(sample):
    img = samples.make(sample, 256, seed=1, epoch=0, index=0, preset="none", train=False)["image"]
    hazy = samples.haze(img, 0.5)
    spread = lambda a: a.reshape(-1, 3).max(0).astype(int) - a.reshape(-1, 3).min(0)   # noqa: E731
    assert hazy.shape == img.shape and (abs(spread(hazy) - 0.5 * spread(img)) <= 1).all()
    assert np.array_equal(samples.haze(img, 0.0), img)
    from scipy import ndimage
    soft = ndimage.gaussian_filter(img, sigma=(2, 2, 0))
    edges = lambda a: np.abs(np.diff(a.astype(int), axis=1)).max()   # noqa: E731
    assert edges(soft) < edges(img)


def test_evaluation_and_the_none_preset_are_not_augmented(sample):
    a = samples.make(sample, 256, seed=1, epoch=0, index=0, train=False)
    b = samples.make(sample, 256, seed=2, epoch=7, index=3, train=False)
    assert np.array_equal(a["image"], b["image"]) and a["boxes"] == b["boxes"]
    assert a["scale"] == pytest.approx(256 / 1200)             # shrunk to fit, top-left, grey below
    assert (a["image"][200:, :] == samples.GREY).all()
    assert [round(v) for v in a["boxes"][0]] == [1, 53, 43, 21, 21]
    assert [round(v) for v in a["ignore"][0]] == [94, 30, 17, 17]
    none = samples.make(sample, 256, seed=5, epoch=1, index=1, preset="none")
    assert np.array_equal(none["image"], a["image"])


def test_a_frame_without_boxes_and_a_rotated_jpeg_load(tmp_path, sample):
    empty = {**sample, "boxes": [], "ignore": []}
    got = samples.make(empty, 256, seed=1, epoch=0, index=0)
    assert got["boxes"] == [] and got["image"].shape == (256, 256, 3)
    # a JPEG with EXIF orientation 6 (stored sideways): sizes are the upright ones (D27)
    a = np.zeros((300, 200, 3), np.uint8)
    a[:, :100] = (250, 0, 0)                                   # stored: left half red
    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(a).save(p, exif=exif, quality=95)
    s = {"image": str(p), "width": 300, "height": 200, "boxes": [], "ignore": []}
    img = samples.make(s, 320, seed=0, epoch=0, index=0, train=False)["image"]
    assert img[150:200, 10].mean() < 60 or img[10:90, 10, 0].mean() > 200   # upright, not sideways
    assert (img[200:, :] == samples.GREY).all() and (img[:200, 300:] == samples.GREY).all()


def test_order_is_a_fixed_shuffle_per_seed_and_epoch():
    a = samples.order(50, seed=1, epoch=0)
    assert sorted(a) == list(range(50)) and a != list(range(50))
    assert a == samples.order(50, seed=1, epoch=0)
    assert a != samples.order(50, seed=1, epoch=1) and a != samples.order(50, seed=2, epoch=0)
    assert samples.DEFAULT_PRESET in samples.PRESETS and "haze" in samples.DEFAULT_PRESET
