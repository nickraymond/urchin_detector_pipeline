"""S3 bite 2: tiles + tile-edge duplicates (D23), sam_raw files and the re-run rule, and the
SAM worker process driven with a fake predictor (tests/fake_sam.py). Synthetic images only."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.sam import classes, results, tiles
from app.stages import label
from tests.fake_sam import COLOURS, run_sam

FAKE = "tests.fake_sam:FakePredictor"


def det(box, region, prompt="urchin", score=0.9):
    x0, y0, x1, y1 = box
    return {"prompt": prompt, "score": score, "box": list(box), "region": region,
            "mask": np.ones((y1 - y0, x1 - x0), bool)}


def test_old_tiles_import_still_works():
    from app.sam_test.tiles import tile_grid
    assert tile_grid is tiles.tile_grid


def test_from_region_moves_mask_to_frame_coordinates():
    m = np.zeros((100, 100), bool)
    m[10:20, 30:35] = True
    d = tiles.from_region(m, 0.8, "urchin", (500, 200, 600, 300))
    assert d["box"] == [530, 210, 535, 220] and d["mask"].shape == (10, 5)
    assert tiles.from_region(np.zeros((4, 4), bool), 0.9, "x", (0, 0, 4, 4)) is None


def test_cut_off_half_at_a_tile_edge_is_dropped_but_whole_one_kept():
    W, H = 2000, 1000
    left, right = (0, 0, 1008, 1000), (800, 0, 2000, 1000)
    whole = det((950, 100, 1050, 200), right)        # inside the right tile
    half = det((950, 100, 1008, 200), left)          # cut by the left tile's right edge
    lone = det((1000, 500, 1008, 520), left)         # cut, but nothing else covers it
    kept = tiles.remove_duplicates([half, whole, lone], W, H)
    assert sorted(d["box"] for d in kept) == sorted([whole["box"], lone["box"]])
    assert tiles.area(next(d for d in kept if d["box"] == whole["box"])) == 100 * 100
    # IoU alone would keep both: 58 % of the pixels of one are the other
    assert tiles.iou(half, whole) < 0.6


def test_an_object_wider_than_the_overlap_is_unioned_across_the_seam():
    W, H = 2000, 1000
    left, right = (0, 0, 1008, 1000), (800, 0, 2000, 1000)
    a = det((700, 100, 1008, 200), left)             # cut by the left tile's right edge
    b = det((800, 100, 1300, 200), right)            # cut by the right tile's left edge
    kept = tiles.remove_duplicates([a, b], W, H)
    assert len(kept) == 1 and kept[0]["box"] == [700, 100, 1300, 200]
    assert tiles.area(kept[0]) == 600 * 100


def test_iou_duplicates_keep_the_higher_score_per_prompt():
    a = det((0, 0, 10, 10), (0, 0, 100, 100), score=0.5)
    b = det((1, 0, 11, 10), (0, 0, 100, 100), score=0.7)
    c = det((1, 0, 11, 10), (0, 0, 100, 100), prompt="fish", score=0.6)
    assert tiles.remove_duplicates([a, b, c], 100, 100) == [b, c]


def test_settings_key_changes_with_prompt_or_setting():
    unit = {"key": "fish", "prompts": ["fish"], "kind": "object"}
    s = classes.parse(classes.DEFAULT_FILE.read_text())["settings"]
    k = results.settings_key(unit, s)
    assert k == results.settings_key(dict(unit), dict(s))
    assert k != results.settings_key({**unit, "prompts": ["fishes"]}, s)
    assert k != results.settings_key(unit, {**s, "confidence": 0.5})
    assert k != results.settings_key({**unit, "kind": "area"}, s)


def test_write_frame_rle_reviewed_masks_and_class_ids(tmp_path):
    units = classes.run_units(classes.parse(classes.DEFAULT_FILE.read_text())["classes"])
    urchin, fish = units[0], next(u for u in units if u["key"] == "fish")
    s = classes.parse(classes.DEFAULT_FILE.read_text())["settings"]
    d = det((2, 3, 6, 5), (0, 0, 20, 10))
    data = results.write_frame(tmp_path, "f.jpg", 20, 10, s, {
        "urchin": {"unit": urchin, "seconds": 1.0, "dets": [d]},
        "fish": {"unit": fish, "seconds": 0.5, "dets": []}})              # zero masks saved too
    m = data["units"]["urchin"]["masks"][0]
    assert m["class_id"] is None and m["area"] == 8 and m["box"] == [2, 3, 6, 5]
    full = results.decode(m["rle"])
    assert full.shape == (10, 20) and full.sum() == 8 and full[3:5, 2:6].all()
    assert data["units"]["fish"]["masks"] == []
    assert results.to_run(tmp_path, "f.jpg", units, s) == [u for u in units
                                                           if u["key"] not in ("urchin", "fish")]
    assert json.loads((tmp_path / "sam_raw" / "class_ids.json").read_text()) == {
        "1": "purple urchin", "2": "red urchin", "10": "black urchin", "6": "fish"}
    # a re-run replaces unreviewed masks and keeps reviewed ones, marked as older
    raw = results.read(tmp_path, "f.jpg")
    raw["units"]["urchin"]["masks"][0]["reviewed"] = True
    results.raw_path(tmp_path, "f.jpg").write_text(json.dumps(raw))
    data = results.write_frame(tmp_path, "f.jpg", 20, 10, {**s, "confidence": 0.5}, {
        "urchin": {"unit": urchin, "seconds": 1.0, "dets": [det((10, 0, 12, 2), (0, 0, 20, 10))]}})
    masks = data["units"]["urchin"]["masks"]
    assert [(m["reviewed"], m["from_older_settings"]) for m in masks] == [(True, True),
                                                                          (False, False)]
    assert data["units"]["fish"]["masks"] == []                           # other units untouched
    # the classes check now holds those names to their ids (D6); a rename keeps the id (D39)
    with pytest.raises(classes.ClassesError):
        classes.save(tmp_path, classes.DEFAULT_FILE.read_text().replace("{id: 6, name: fish,",
                                                                        "{id: 11, name: fish,"))


def synthetic_frame(path, size=(2600, 1200)):
    """Two urchins (one straddling the first tile seam), a sea star, a kelp band."""
    im = Image.new("RGB", size, (10, 60, 90))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1000, size[0], size[1]), fill=COLOURS["kelp"])
    d.ellipse((960, 300, 1060, 400), fill=COLOURS["urchin"])   # across x = 1008
    d.ellipse((200, 200, 260, 260), fill=COLOURS["urchin"])
    d.ellipse((2000, 500, 2100, 600), fill=COLOURS["sea star"])
    im.save(path)
    return path


def project_with_frames(tmp_path, n=2):
    folder = tmp_path / "p"
    (folder / "frames").mkdir(parents=True)
    for i in range(n):
        synthetic_frame(folder / "frames" / f"v_aaaaaa_{i}.png")
    return folder, [f"v_aaaaaa_{i}.png" for i in range(n)]


def test_worker_process_labels_frames_with_tiles_and_skips_done_units(tmp_path):
    folder, files = project_with_frames(tmp_path)
    out = run_sam(folder, files)
    assert [e["i"] for e in out["frames"]] == [1, 2] and out["code"] == 0 and out["done"]
    assert [e["name"] for e in out["frames"]] == files and out["frames"][0]["counts"]["urchin"] == 2
    raw = results.read(folder, files[0])
    assert (raw["width"], raw["height"]) == (2600, 1200)
    counts = {k: len(u["masks"]) for k, u in raw["units"].items()}
    assert counts["urchin"] == 2                       # the seam urchin counted once (D23)
    assert counts["sea star"] == 1 and counts["kelp"] == 1 and counts["fish"] == 0
    assert len(counts) == 8                            # every unit saved, also with no masks
    assert (folder / "classes.yaml").is_file()        # copied in on first use
    # same settings again: nothing re-runs
    again = run_sam(folder, files)
    assert all(e.get("skipped") for e in again["frames"])
    # a changed prompt re-runs that unit only
    text = (folder / "classes.yaml").read_text().replace("prompts: [fish]", "prompts: [rock]")
    classes.save(folder, text)
    before = results.read(folder, files[0])["units"]["urchin"]["run_at"]
    third = run_sam(folder, files[:1])
    assert list(third["frames"][0]["masks"]) == ["fish"]
    assert results.read(folder, files[0])["units"]["urchin"]["run_at"] == before


def test_kelp_area_and_min_mask_area(tmp_path):
    folder, files = project_with_frames(tmp_path, n=1)
    text = classes.DEFAULT_FILE.read_text().replace("min_mask_area_px: 500 ",
                                                    "min_mask_area_px: 5000")
    classes.save(folder, text)
    run_sam(folder, files)
    raw = results.read(folder, files[0])
    assert len(raw["units"]["urchin"]["masks"]) == 1   # the 60 px urchin is below 5000 px
    assert raw["units"]["kelp"]["masks"][0]["class_id"] == 9


def test_a_failing_worker_reports_and_keeps_finished_frames(tmp_path):
    folder, files = project_with_frames(tmp_path, n=1)
    (folder / "frames" / "bad.png").write_bytes(b"not an image")
    out = run_sam(folder, files + ["bad.png"])
    assert out["code"] != 0 and out["done"] is None and len(out["frames"]) == 1
    assert "Traceback" in out["log"]
    assert results.read(folder, files[0]) is not None


def test_a_missing_frame_file_is_named(tmp_path):
    folder, files = project_with_frames(tmp_path, n=1)
    with pytest.raises(label.LabelError, match="v_gone_1.png is missing"):
        label.build_job({"folder": str(folder), "files": files + ["v_gone_1.png"]})
