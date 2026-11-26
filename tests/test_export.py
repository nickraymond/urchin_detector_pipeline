"""S5 bite 2: export writers and checks (D41, D43, D44). Synthetic frames and masks only."""

import csv
import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from app import db
from app.export import checks, splits, writers
from app.review import state
from app.sam import classes, results

PARSED = classes.parse(classes.DEFAULT_FILE.read_text())
CLASSES, SETTINGS = PARSED["classes"], PARSED["settings"]
UNITS = {u["key"]: u for u in classes.run_units(CLASSES)}
W, H = 80, 60


# ---- polygons ----

def raster(polys, w, h):
    rles = mask_utils.frPyObjects(polys, h, w)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_polygon_of_a_disc_rasterises_back_to_the_mask():
    yy, xx = np.mgrid[0:60, 0:80]
    disc = (xx - 40) ** 2 + (yy - 30) ** 2 <= 20 ** 2
    polys = writers.polygons(disc[5:56, 15:66], 15, 5)
    assert len(polys) == 1 and len(polys[0]) < 0.5 * 2 * 4 * 41   # simplified, not every step
    assert iou(raster(polys, 80, 60), disc) > 0.95


def test_polygons_of_two_pieces_and_a_ring_fill_the_hole():
    m = np.zeros((30, 40), bool)
    m[2:10, 2:10] = True                      # a square
    m[15:28, 15:38] = True
    m[19:24, 20:30] = False                   # a block with a hole
    polys = writers.polygons(m, 0, 0)
    assert len(polys) == 2                    # two outlines, the hole dropped
    back = raster(polys, 40, 30)
    filled = m.copy()
    filled[19:24, 20:30] = True
    assert iou(back, filled) > 0.95
    assert writers.polygons(np.zeros((5, 5), bool), 0, 0) == []


def test_a_one_pixel_mask_still_gets_an_outline():
    m = np.zeros((3, 3), bool)
    m[1, 1] = True
    (p,) = writers.polygons(m, 10, 20)
    assert len(p) >= 6 and raster([p], 40, 40)[21, 11]


# ---- splits ----

def test_site_and_dive_date_from_the_name_or_folder():
    assert splits.site_date("/v/Some-Site-North-25-10-16-1130-x-c1.MP4") == "Some-Site-North 2025-10-16"
    assert splits.site_date("/v/Reef-25-10-18/GX010001.MP4") == "Reef 2025-10-18"
    assert splits.site_date("/v/GX010001.MP4") is None


def test_group_assignment_is_deterministic_close_to_70_20_10_and_never_empty():
    groups = {f"g{i}": n for i, n in enumerate([40, 30, 10, 10, 5, 5])}
    a = splits.assign(groups)
    assert a == splits.assign(dict(reversed(list(groups.items()))))
    per = {s: sum(n for g, n in groups.items() if a[g] == s) for s in splits.SPLITS}
    assert per == {"train": 70, "valid": 20, "test": 10}
    a3 = splits.assign({"a": 10, "b": 10, "c": 10})
    assert sorted(a3.values()) == ["test", "train", "valid"]


def test_fewer_than_three_groups_split_by_video_with_a_warning():
    frames = [{"file": f"f{i}.jpg", "source": f"/v/Site-25-10-16-09{i % 4}0-x.MP4"}
              for i in range(8)]
    sp = splits.split(frames)                 # one site + date, four videos
    assert sp["grouped_by"] == "video" and "fair split" in sp["warning"]
    assert set(sp["by_frame"].values()) == set(splits.SPLITS)
    by_video = {}
    for f in frames:
        by_video.setdefault(f["source"], set()).add(sp["by_frame"][f["file"]])
    assert all(len(v) == 1 for v in by_video.values())   # a video never spans two splits


# ---- the whole export ----

def det(box, score=0.9):
    x0, y0, x1, y1 = box
    return {"prompt": "p", "score": score, "box": list(box),
            "mask": np.ones((y1 - y0, x1 - x0), bool)}


def make_frame(folder, conn, name, source, rotate=False):
    img = Image.new("RGB", (H, W) if rotate else (W, H), (20, 60, 90))
    if rotate:          # stored sideways; EXIF 6 turns it upright to W x H
        ex = Image.Exif()
        ex[0x0112] = 6
        img.save(folder / "frames" / name, quality=95, exif=ex)
    else:
        img.save(folder / "frames" / name, quality=95)
    conn.execute("INSERT INTO frames (file, video, timestamp_ms, status) VALUES (?, ?, ?, 'kept')",
                 (name, source, 1000))
    conn.commit()
    for unit, dets in (("urchin", [det((10, 10, 20, 20)), det((30, 10, 42, 22), 0.8)]),
                       ("sea star", [det((50, 30, 60, 40), 0.7)]),
                       ("kelp", [det((0, 40, 40, 60))]),
                       ("rock/substrate", [det((0, 30, 80, 60))])):
        results.write_frame(folder, name, W, H, SETTINGS,
                            {unit: {"unit": UNITS[unit], "seconds": 0.1, "dets": dets}})


@pytest.fixture
def proj(tmp_path):
    (tmp_path / "frames").mkdir()
    conn = db.connect(tmp_path / "state.sqlite", create=True)
    classes.save(tmp_path, classes.DEFAULT_FILE.read_text())
    sites = ["A", "A", "B", "C", "D", "E"]
    for i, s in enumerate(sites):
        f = f"f{i}.jpg"
        make_frame(tmp_path, conn, f, f"/v/Site{s}-25-10-1{i}-0900-x.MP4", rotate=(i == 1))
        state.set_class(conn, tmp_path, f, [1], 1, CLASSES)       # purple
        state.set_class(conn, tmp_path, f, [2], 2, CLASSES)       # red
        if i < 5:
            state.accept(conn, tmp_path, f, CLASSES)              # f5 stays un-accepted
    state.set_split(conn, "f2.jpg", False)
    state.set_checked(conn, "f0.jpg", 1, True, CLASSES)
    state.delete(conn, tmp_path, "f3.jpg", [3])                   # no sea star on f3
    yield tmp_path, conn
    conn.close()


def test_export_writes_every_output_and_the_checks_pass(proj):
    folder, conn = proj
    res = writers.run(folder, conn)
    assert res["ok"], res["checks"]
    out = folder / "export"
    s = res["summary"]
    assert s["frames"] == 5 and s["coco_images"] == 4 and s["split_no"] == 1
    assert sorted(p.name for p in (out / "images").iterdir()) == [f"f{i}.jpg" for i in range(5)]
    # D44: plain frames copied byte for byte; the rotated one re-saved upright
    assert (out / "images" / "f0.jpg").read_bytes() == (folder / "frames" / "f0.jpg").read_bytes()
    with Image.open(out / "images" / "f1.jpg") as im:
        assert im.size == (W, H) and im.getexif().get(0x0112, 1) == 1
    coco = json.loads((out / "detector_coco.json").read_text())
    assert {c["name"] for c in coco["categories"]} >= {"purple urchin", "red urchin", "sea star"}
    assert "kelp" not in {c["name"] for c in coco["categories"]}             # objects only
    assert "f2.jpg" not in {i["file_name"] for i in coco["images"]}          # split check "no"
    f0 = next(i for i in coco["images"] if i["file_name"] == "f0.jpg")
    assert f0["checked_classes"] == [1]
    anns = [a for a in coco["annotations"] if a["image_id"] == f0["id"]]
    assert sorted(a["category_id"] for a in anns) == [1, 2, 3]
    assert next(a for a in anns if a["category_id"] == 1)["bbox"] == [10, 10, 10, 10]
    m = np.asarray(Image.open(out / "seg_masks" / "f0.png"))
    assert m.shape == (H, W) and m[15, 15] == 1 and m[15, 35] == 2 and m[35, 55] == 3
    assert m[50, 10] == 9 and m[35, 70] == 7 and m[5, 5] == 0 and 255 not in m
    ids = [c["id"] for c in json.loads((out / "seg_masks" / "classes.json").read_text())["classes"]]
    assert ids[0] == 0 and set(ids) >= {1, 2, 3, 7, 9}
    rows = {r["frame"]: r for r in csv.DictReader((out / "frame_metrics.csv").open())}
    assert set(rows) == {f"f{i}.jpg" for i in range(5)}
    assert rows["f2.jpg"]["split_ok"] == "no" and rows["f2.jpg"]["purple urchin count"] == "1"
    assert rows["f3.jpg"]["sea star count"] == "0"
    assert float(rows["f0.jpg"]["purple urchin cover_pct"]) == pytest.approx(100 * 100 / (W * H), abs=1e-4)
    assert float(rows["f0.jpg"]["largest_object_pct"]) == pytest.approx(100 * 144 / (W * H), abs=1e-4)
    listed = [ln for s in splits.SPLITS for ln in (out / "splits" / f"{s}.txt").read_text().split()]
    assert sorted(listed) == [f"f{i}.jpg" for i in range(5)]
    assert all(c["ok"] for c in json.loads((out / "checks.json").read_text()))


def test_re_export_rebuilds_from_scratch(proj):
    folder, conn = proj
    writers.run(folder, conn)
    (folder / "export" / "stray.txt").write_text("x")
    state.reopen(conn, "f4.jpg")
    res = writers.run(folder, conn)
    assert res["ok"] and res["summary"]["frames"] == 4
    assert not (folder / "export" / "stray.txt").exists()
    assert not (folder / "export" / "images" / "f4.jpg").exists()
    assert not (folder / "export.new").exists() and not (folder / "export.old").exists()


def test_an_untagged_urchin_on_an_accepted_frame_stops_the_export_and_keeps_the_old(proj):
    folder, conn = proj
    writers.run(folder, conn)
    results.write_frame(folder, "f0.jpg", W, H, {**SETTINGS, "confidence": 0.5},
                        {"urchin": {"unit": UNITS["urchin"], "seconds": 0.1,
                                    "dets": [det((60, 0, 70, 8))]}},
                        state.reviewed_ids(folder, "f0.jpg"))       # a re-run after accepting
    with pytest.raises(writers.ExportError, match="untagged"):
        writers.run(folder, conn)
    assert (folder / "export" / "detector_coco.json").is_file()
    assert not (folder / "export.new").exists()


def test_checks_catch_broken_files(proj):
    folder, conn = proj
    writers.run(folder, conn)
    out = folder / "export"
    Image.new("L", (10, 10)).save(out / "seg_masks" / "f0.png")             # wrong size
    Image.fromarray(np.full((H, W), 42, np.uint8), mode="L").save(out / "seg_masks" / "f3.png")
    coco = json.loads((out / "detector_coco.json").read_text())
    coco["annotations"][0]["bbox"] = [70, 50, 20, 20]                         # outside
    (out / "detector_coco.json").write_text(json.dumps(coco))
    (out / "images" / "._f0.jpg").write_bytes(b"\0")           # macOS exFAT sidecar: ignored
    (out / "splits" / "valid.txt").write_text("f0.jpg\n" + (out / "splits" / "valid.txt").read_text())
    found = {c["name"]: c for c in checks.run_all(out)}
    assert not found["mask sizes and ids"]["ok"]
    assert "f0.png: 10x10" in found["mask sizes and ids"]["detail"]
    assert "ids [42]" in found["mask sizes and ids"]["detail"]
    assert not found["COCO loads with pycocotools"]["ok"]
    assert not found["metrics match masks and COCO"]["ok"]
    assert found["images present"]["ok"]
    assert "f0.jpg: in" in found["every image in one split"]["detail"]          # two splits
    assert "._f0" not in found["every image in one split"]["detail"]
    (out / "detector_coco.json").write_text("{not json")
    assert not {c["name"]: c for c in checks.run_all(out)}["COCO loads with pycocotools"]["ok"]


def test_a_failed_check_keeps_the_previous_export(proj, monkeypatch):
    folder, conn = proj
    writers.run(folder, conn)
    before = (folder / "export" / "detector_coco.json").read_bytes()
    monkeypatch.setattr(checks, "CHECKS", (lambda out: {"name": "x", "ok": False, "detail": "d"},))
    state.reopen(conn, "f4.jpg")
    res = writers.run(folder, conn)
    assert not res["ok"] and res["folder"].endswith("export.failed")
    assert (folder / "export" / "detector_coco.json").read_bytes() == before
