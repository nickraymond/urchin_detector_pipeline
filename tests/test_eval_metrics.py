"""S9 bite 2a (D91): the figures on hand-made frames, and AP50 against pycocotools' COCOeval."""

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
    assert c["within_20"] == {"value": 1.0, "frames": 2, "objects": 13}
    assert c["within_20_strict"]["value"] == 0.5 and c["passed_by_one_rule"]["value"] == 1
    assert c["count_error_mean"]["value"] == pytest.approx(-0.5) and c["count_error_mean_abs"]["value"] == 1.5
    assert c["empty_with_pred"] == {"value": 0.5, "frames": 2}


def test_precision_recall_small_objects_and_largest_box():
    frames = [fr("a", [[1, 0, 0, 40, 40], [1, 60, 60, 10, 10]]), fr("b", [[1, 10, 10, 20, 20]])]
    preds = {"a": [[1, 0, 0, 40, 40, .9], [1, 80, 0, 10, 10, .8]],      # the big one found, a false one
             "b": [[1, 11, 11, 20, 20, .7]]}
    c = metrics.figures(frames, preds, [1], 2.0, 0.3)["classes"]["1"]
    assert c["precision"]["value"] == pytest.approx(2 / 3) and c["recall"]["value"] == pytest.approx(2 / 3)
    # small at the eval scale: shorter side x 2 < 32 -> the 10 px box only, not found
    assert c["small_recall"] == {"value": 0.0, "frames": 2, "objects": 1}
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
