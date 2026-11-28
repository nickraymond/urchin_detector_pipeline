"""S9 bite 2 (D91, D92): the figures on hand-made frames, AP50 against pycocotools' COCOeval, the intervals and the
rank floor."""

import contextlib
import io

import numpy as np
import pytest

from app.evaluate import metrics


def fr(file, boxes, w=100, h=100):
    return {"file": file, "width": w, "height": h, "boxes": boxes}


def test_count_rules_and_empty_frames():
    frames = [fr("a", [[1, 0, 0, 10, 10]] * 3), fr("b", [[1, 50, 50, 10, 10]] * 10), fr("c", []), fr("d", [])]
    preds = {"a": [[1, 0, 0, 10, 10, .9]] * 4,                           # 4 for 3: off by one (20 % of 3 < 1)
             "b": [[1, 50, 50, 10, 10, .9]] * 8,                         # 8 for 10: off by two = 20 %
             "c": [[1, 5, 5, 5, 5, .9]], "d": [[1, 5, 5, 5, 5, .1]]}     # d's is under the 0.3 threshold
    c = metrics.figures(frames, preds, [1], 1.0, 0.3)["classes"]["1"]
    assert c["within_20"] == {"value": 1.0, "frames": 2, "objects": 13, "interval": [0.34238, 1.0]}   # Wilson, 2 of 2
    assert c["within_20_strict"]["value"] == 0.5 and c["passed_by_one_rule"]["value"] == 1
    assert c["count_error_mean"]["value"] == pytest.approx(-0.5) and c["count_error_mean_abs"]["value"] == 1.5
    assert c["empty_with_pred"] == {"value": 0.5, "frames": 2, "interval": metrics.wilson(1, 2)}
    assert c["passed_by_one_rule"]["interval"] is None


def test_precision_recall_small_objects_and_largest_box():
    frames = [fr("a", [[1, 0, 0, 40, 40], [1, 60, 60, 10, 10]]), fr("b", [[1, 10, 10, 20, 20]])]
    preds = {"a": [[1, 0, 0, 40, 40, .9], [1, 80, 0, 10, 10, .8]],      # the big one found, a false one
             "b": [[1, 11, 11, 20, 20, .7]]}
    c = metrics.figures(frames, preds, [1], 2.0, 0.3)["classes"]["1"]
    assert c["precision"]["value"] == pytest.approx(2 / 3) and c["recall"]["value"] == pytest.approx(2 / 3)
    # small at the eval scale: shorter side x 2 < 32 -> the 10 px box only, not found
    assert {k: c["small_recall"][k] for k in ("value", "frames", "objects")} == {"value": 0.0, "frames": 2, "objects": 1}
    # a: 16 % against 16 % -> 0; b: 4 % against 4 % -> 0
    assert c["largest_pct_error"]["value"] == pytest.approx(0.0)
    none = metrics.figures(frames, {}, [1], 2.0, 0.3)["classes"]["1"]
    assert none["precision"]["value"] is None and none["recall"]["value"] == 0 and none["largest_pct_error"]["value"] == 10


def test_ap50_matches_pycocotools_on_a_hand_made_case():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rng = np.random.default_rng(5)
    frames, preds, gt, dt = [], {}, [], []
    for i in range(12):
        boxes, ps = [], []
        for k in range(int(rng.integers(0, 5))):
            x, y, s = (float(v) for v in rng.integers(0, 150, 3))
            s += 8
            boxes.append([1, x, y, s, s])
            if rng.random() < 0.7:                                   # found, a little off
                ps.append([1, x + rng.normal(0, 2), y + rng.normal(0, 2), s, s, float(rng.random())])
        for _ in range(int(rng.integers(0, 3))):                    # false ones
            ps.append([1, *(float(v) for v in rng.integers(0, 150, 2)), 20.0, 20.0, float(rng.random())])
        frames.append(fr(f"f{i}", boxes, 200, 200))
        preds[f"f{i}"] = ps
        gt += [{"id": len(gt) + k + 1, "image_id": i + 1, "category_id": 1, "bbox": b[1:], "area": b[3] * b[4],
                "iscrowd": 0} for k, b in enumerate(boxes)]                 # one id per box
        dt += [{"image_id": i + 1, "category_id": 1, "bbox": p[1:5], "score": p[5]} for p in ps]
    coco = COCO()
    coco.dataset = {"images": [{"id": i + 1, "width": 200, "height": 200} for i in range(12)], "annotations": gt,
                    "categories": [{"id": 1, "name": "x"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(dt), "bbox")
        ev.params.iouThrs = np.array([0.5])
        ev.params.maxDets = [100]
        ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
        ev.evaluate()
        ev.accumulate()
    p = ev.eval["precision"][0, :, 0, 0, 0]
    theirs = float(np.mean(p[p > -1]))
    ours = metrics.figures(frames, preds, [1], 1.0, 0.3)["classes"]["1"]["ap50"]["value"]
    assert ours == pytest.approx(theirs, abs=1e-6) and 0 < ours < 1


def test_wilson_matches_scipy():
    # the values scipy 1.17's binomtest(k, n).proportion_ci(method="wilson") gives (read 2026-11-27)
    assert metrics.wilson(0, 10) == pytest.approx([0.0, 0.27753], abs=1e-5)
    assert metrics.wilson(5, 10) == pytest.approx([0.23659, 0.76341], abs=1e-5)
    assert metrics.wilson(7, 13) == pytest.approx([0.29144, 0.76794], abs=1e-5)
    assert metrics.wilson(0, 0) is None
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((1, 2), (3, 17), (40, 41)):
        ci = stats.binomtest(k, n).proportion_ci(method="wilson")
        assert metrics.wilson(k, n) == pytest.approx([ci.low, ci.high], abs=1e-6)


def _set(groups_of, found_in):
    """One true box per frame; a prediction on it in the frames of found_in's groups."""
    frames, preds = [], {}
    for i, g in enumerate(groups_of):
        frames.append(fr(f"f{i}", [[1, 10, 10, 20, 20]]))
        preds[f"f{i}"] = [[1, 10, 10, 20, 20, .9]] if g in found_in else []
    return frames, preds, {f"f{i}": g for i, g in enumerate(groups_of)}


def test_the_bootstrap_is_seeded_and_holds_the_value():
    frames, preds, _ = _set([i % 3 for i in range(30)], {0, 1})
    a = metrics.figures(frames, preds, [1], 1.0, 0.3)
    b = metrics.figures(frames, preds, [1], 1.0, 0.3)
    r = a["classes"]["1"]["recall"]
    assert r == b["classes"]["1"]["recall"] and r["interval"][0] < r["value"] < r["interval"][1]
    assert a["intervals"]["unit"] == "frames" and a["intervals"]["draws"] == 1000 and "dives only" in a["intervals"]["note"]
    assert a["classes"]["1"]["precision"]["interval"] == [1.0, 1.0]          # every prediction is right in every draw
    assert a["classes"]["1"]["ap50"]["interval"][1] <= 1.0


def test_from_five_groups_whole_groups_are_resampled_and_the_interval_is_wider():
    groups_of = [i // 4 for i in range(40)]                                 # 10 dives of 4 frames
    frames, preds, groups = _set(groups_of, {0, 1, 2, 3, 4})                # half the dives found, half missed
    by_frame = metrics.figures(frames, preds, [1], 1.0, 0.3)
    by_group = metrics.figures(frames, preds, [1], 1.0, 0.3, groups)
    assert by_frame["intervals"]["unit"] == "frames" and by_group["intervals"]["unit"] == "groups"
    assert by_group["intervals"]["groups"] == 10 and "note" not in by_group["intervals"]
    lo_f, hi_f = by_frame["classes"]["1"]["recall"]["interval"]
    lo_g, hi_g = by_group["classes"]["1"]["recall"]["interval"]
    assert hi_g - lo_g > hi_f - lo_f                                        # frames of one dive move together
    for idx in metrics.draws([f["file"] for f in frames], groups, n=20)["index"]:
        counts = np.bincount([groups_of[i] for i in idx], minlength=10)
        assert all(n % 4 == 0 for n in counts)                              # whole dives only


def test_fewer_than_five_groups_resample_frames_and_say_so():
    frames, preds, groups = _set([i % 4 for i in range(20)], {0, 1})
    d = metrics.figures(frames, preds, [1], 1.0, 0.3, groups)["intervals"]
    assert d["unit"] == "frames" and d["groups"] == 4 and "4 site-and-date group(s)" in d["note"]


def test_the_rank_floor():
    nine, p9, _ = _set([0] * 9, {0})
    floor = metrics.figures(nine, p9, [1], 1.0, 0.3)["classes"]["1"]["rank_floor"]
    assert not floor["ok"] and "9 test frames" in floor["why"]
    frames = [fr(f"f{i}", [[1, 20 * j, 0, 10, 10] for j in range(3)]) for i in range(10)]     # 10 frames, 30 objects
    assert metrics.figures(frames, {}, [1], 1.0, 0.3)["classes"]["1"]["rank_floor"] == {"ok": True, "why": ""}
    few = metrics.figures(frames[:10], {}, [1, 2], 1.0, 0.3)["classes"]["2"]["rank_floor"]
    assert not few["ok"] and "0 true objects" in few["why"]
