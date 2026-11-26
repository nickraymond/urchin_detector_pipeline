"""S5 bite 7 (D45): areas on the whole frame for tiled frames, so they have no tile seams.
Fake predictor and a synthetic frame only."""

from PIL import Image, ImageDraw

from app.sam import classes, results
from app.sam.worker import label_frame
from tests.fake_sam import COLOURS, FakePredictor

PARSED = classes.parse(classes.DEFAULT_FILE.read_text())
UNITS = {u["key"]: u for u in classes.run_units(PARSED["classes"])}


def frame():
    """3000 x 1400: a rock blob across a tile seam, and an urchin."""
    im = Image.new("RGB", (3000, 1400), (10, 60, 90))
    d = ImageDraw.Draw(im)
    d.ellipse((600, 500, 1700, 1300), fill=COLOURS["rock"])
    d.ellipse((2200, 200, 2300, 300), fill=COLOURS["urchin"])
    return im


def test_areas_run_once_on_the_scaled_whole_frame_objects_still_in_tiles():
    s = PARSED["settings"]
    assert s["areas_whole_px"] == 2016
    pred = FakePredictor({})
    out = label_frame(pred, frame(), [UNITS["urchin"], UNITS["rock/substrate"]], s)
    (rock,) = out["rock/substrate"]["dets"]                 # one piece, no tile cuts
    assert rock["region"] == (0, 0, 3000, 1400)
    x0, y0, x1, y1 = rock["box"]
    assert abs(x0 - 600) <= 2 and abs(x1 - 1701) <= 2 and abs(y0 - 500) <= 2 and abs(y1 - 1301) <= 2
    assert rock["mask"].shape == (y1 - y0, x1 - x0)
    assert len(out["urchin"]["dets"]) == 1                  # objects: from the tiles, as before
    n_tiles = pred.calls - 1
    assert n_tiles > 1                                       # 1 whole look + the tiles


def test_zero_keeps_areas_in_tiles_and_object_keys_never_change():
    s0 = {**PARSED["settings"], "areas_whole_px": 0}
    pred = FakePredictor({})
    out = label_frame(pred, frame(), [UNITS["rock/substrate"]], s0)
    assert all(d["region"] != (0, 0, 3000, 1400) for d in out["rock/substrate"]["dets"])
    # the setting is in the area units' settings key only, so turning it on re-runs areas only
    s = PARSED["settings"]
    assert results.settings_key(UNITS["urchin"], s) == results.settings_key(UNITS["urchin"], s0)
    assert results.settings_key(UNITS["kelp"], s) != results.settings_key(UNITS["kelp"], s0)
    old = {k: v for k, v in s0.items() if k != "areas_whole_px"}   # an S3 / S4 file
    assert results.settings_key(UNITS["kelp"], old) == results.settings_key(UNITS["kelp"], s0)


def test_small_frames_are_whole_anyway_and_never_scaled_up():
    im = Image.new("RGB", (1600, 900), (10, 60, 90))
    ImageDraw.Draw(im).rectangle((0, 700, 1600, 900), fill=COLOURS["kelp"])
    pred = FakePredictor({})
    out = label_frame(pred, im, [UNITS["kelp"]], PARSED["settings"])
    assert pred.calls == 1 and out["kelp"]["dets"][0]["box"] == [0, 700, 1600, 900]
