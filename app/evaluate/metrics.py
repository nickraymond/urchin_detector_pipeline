"""The figures of one eval (S9 bite 2a, D91): from the test frames and predictions.jsonl, per class. Every figure is
{"value", "frames", "objects", "interval"}: the frames and true objects behind it and its 95 % interval (D92, bite 2b).

At the score threshold (THRESHOLD, the trainer's validation one):
    count error         predicted - true per frame; mean signed and mean absolute, frames with any object of the class
    within_20           frames with >= 1 true object whose |error| <= max(1, 20 % of the true count)
    within_20_strict    the same with 20 % only (the bake-off bar); passed_by_one_rule = the difference
    empty_with_pred     frames with no true object of the class that have a prediction of it (their own figure)
    precision, recall   IoU 0.5, matched greedily by score, same class (the trainer's rule)
    small_recall        recall on true boxes whose shorter side is under 32 px at the eval scale
    largest_pct_error   the largest box's share of the frame, predicted minus true, in percentage points
Over all scores:
    ap50                COCO's 101-point interpolated AP at IoU 0.5 (checked against pycocotools in the tests)

Intervals (D92): Wilson for the shares whose unit is the frame (within_20, within_20_strict, empty_with_pred); a
seeded bootstrap (DRAWS draws, percentiles 2.5 and 97.5) for the rest, resampling frames, or the test split's
site-and-date groups once it has GROUPS_MIN of them, since frames of one dive succeed or fail together. The same
draws serve every class and figure (`draws`), so bite 5's paired comparison can reuse them. passed_by_one_rule is
a count, with no interval. The rank floor: a class's figures may rank two models only with RANK_FRAMES test frames
and RANK_OBJECTS true objects of the class; the paired rule itself is the compare page's (bite 5).
"""

from __future__ import annotations

import numpy as np

SMALL_PX = 32        # D91: 4 cells of YOLOX's finest stride (8 px)
IOU = 0.5
DRAWS, SEED, GROUPS_MIN = 1000, 0, 5      # D92
RANK_FRAMES, RANK_OBJECTS = 10, 30        # D92: the rank floor
Z95 = 1.959964
WILSON = ("within_20", "within_20_strict", "empty_with_pred")


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of [x, y, w, h] boxes (top-left): (len(a), len(b))."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-12)


def match(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pred [x, y, w, h, score], truth [x, y, w, h], one class, one frame. Predictions in score order (highest
    first) each take the unmatched true box with the highest IoU >= 0.5. Returns (prediction matched, true box found)."""
    order = np.argsort(-pred[:, 4], kind="stable") if len(pred) else np.zeros(0, int)
    iou, used, hit = _iou(pred[order, :4], truth), set(), np.zeros(len(pred), bool)
    for k, i in enumerate(order):
        best, at = IOU - 1e-12, -1
        for j in range(len(truth)):
            if j not in used and iou[k, j] >= best:
                best, at = iou[k, j], j
        if at >= 0:
            used.add(at)
            hit[i] = True
    found = np.zeros(len(truth), bool)
    found[list(used)] = True
    return hit, found


def ap50(scored, n_true: int) -> float | None:
    """COCO's AP at one IoU: precision made non-increasing, read at 101 recall points. scored: (score, hit) pairs."""
    if not n_true:
        return None
    scored = np.asarray(scored, float).reshape(-1, 2)
    hits = scored[np.argsort(-scored[:, 0], kind="stable"), 1] > 0
    tp = np.cumsum(hits).astype(float)
    fp = np.cumsum(~hits).astype(float)
    rec, prec = tp / n_true, tp / np.maximum(tp + fp, 1e-12)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    points = np.linspace(0, 1, 101)
    at = np.searchsorted(rec, points, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in at]))


def wilson(k: int, n: int) -> list[float] | None:
    """The 95 % Wilson score interval of k successes in n."""
    if not n:
        return None
    p, z2 = k / n, Z95 * Z95
    mid, half = (p + z2 / (2 * n)) / (1 + z2 / n), Z95 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [round(float(max(0.0, mid - half)), 6), round(float(min(1.0, mid + half)), 6)]


def draws(files: list[str], groups: dict[str, str] | None = None, n: int = DRAWS, seed: int = SEED) -> dict:
    """The bootstrap's draws: frame indices, resampled by frame, or by group when there are GROUPS_MIN groups.
    groups: file -> site-and-date group (None when the version has none, e.g. split by video)."""
    rng = np.random.default_rng(seed)
    names = sorted({groups[f] for f in files}) if groups and all(f in groups for f in files) else []
    out = {"draws": n, "seed": seed, "groups": len(names) or None}
    if len(names) >= GROUPS_MIN:
        members = [np.array([i for i, f in enumerate(files) if groups[f] == g]) for g in names]
        out["unit"] = "groups"
        out["index"] = [np.concatenate([members[j] for j in rng.integers(0, len(names), len(names))])
                        for _ in range(n)]
    else:
        out["unit"] = "frames"
        out["index"] = [rng.integers(0, len(files), len(files)) for _ in range(n)] if files else []
        out["note"] = (f"The test split holds {len(names)} site-and-date group(s), fewer than {GROUPS_MIN}: the "
                       "intervals resample frames and speak for these dives only." if names else
                       "The test split's frames have no site-and-date groups (the version was split by video or "
                       "the groups are unknown): the intervals resample frames and speak for these dives only.")
    return out


def _stats(frames: list[dict], predictions: dict[str, list], c: int, factor: float | None, threshold: float) -> dict:
    """Per frame, for class c: everything a figure sums or averages (arrays over the frames)."""
    rows, scored = [], []
    for f in frames:
        t = np.array([b[1:5] for b in f["boxes"] if b[0] == c], float).reshape(-1, 4)
        p_all = np.array([b[1:6] for b in predictions.get(f["file"], []) if b[0] == c], float).reshape(-1, 5)
        p = p_all[p_all[:, 4] >= threshold]
        hit_all, _ = match(p_all, t)
        scored.append(np.column_stack([p_all[:, 4], hit_all]).reshape(-1, 2))
        hit, found = match(p, t)
        n_t, e = len(t), len(p) - len(t)
        small = np.minimum(t[:, 2], t[:, 3]) * (factor or 1.0) < SMALL_PX
        area = f["width"] * f["height"]
        largest = ((100 * (p[:, 2] * p[:, 3]).max() / area if len(p) else 0.0)
                   - 100 * (t[:, 2] * t[:, 3]).max() / area) if n_t else np.nan
        rows.append([n_t, len(p), int(hit.sum()), e, abs(e) <= max(1, 0.2 * n_t), abs(e) <= 0.2 * n_t,
                     int(small.sum()), int((found & small).sum()), largest, bool(len(p))])
    a = np.array(rows, float).reshape(-1, 10)
    keys = ("n_true", "n_pred", "tp", "err", "within", "strict", "small_true", "small_hit", "largest", "any_pred")
    return {**{k: a[:, i] for i, k in enumerate(keys)}, "scored": scored}


def _values(st: dict, idx: np.ndarray) -> dict:
    """Every bootstrapped figure's value on the frames idx (with repeats)."""
    n_t = st["n_true"][idx]
    has = n_t > 0
    tp, n_pred, n_true, small = st["tp"][idx].sum(), st["n_pred"][idx].sum(), n_t.sum(), st["small_true"][idx].sum()
    err = st["err"][idx][has]
    scored = np.concatenate([st["scored"][i] for i in idx]) if len(idx) else np.zeros((0, 2))
    return {"count_error_mean": err.mean() if has.any() else None,
            "count_error_mean_abs": np.abs(err).mean() if has.any() else None,
            "precision": tp / n_pred if n_pred else None, "recall": tp / n_true if n_true else None,
            "ap50": ap50(scored, int(n_true)),
            "small_recall": st["small_hit"][idx].sum() / small if small else None,
            "largest_pct_error": np.abs(st["largest"][idx][has]).mean() if has.any() else None}


def _fig(value, frames, objects=None, interval=None) -> dict:
    out = {"value": None if value is None else round(float(value), 6), "frames": int(frames), "interval": interval}
    if objects is not None:
        out["objects"] = int(objects)
    return out


def _percentiles(vals: list) -> list[float] | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def figures(frames: list[dict], predictions: dict[str, list], class_ids: list[int], factor: float | None,
            threshold: float, groups: dict[str, str] | None = None) -> dict:
    """frames: the test samples ({"file", "width", "height", "boxes": [[class id, x, y, w, h]]}); predictions:
    file -> [[class id, x, y, w, h, score]] (native pixels); groups: file -> site-and-date group, or None.
    Returns {"classes": {id: {figure: {...}, "rank_floor": {...}}}, "intervals": {...}}."""
    boot = draws([f["file"] for f in frames], groups)
    out = {}
    for c in class_ids:
        st = _stats(frames, predictions, c, factor, threshold)
        has = st["n_true"] > 0
        n_frames, n_true, small_true = int(has.sum()), int(st["n_true"].sum()), int(st["small_true"].sum())
        within, strict = int(st["within"][has].sum()), int(st["strict"][has].sum())
        empty_n, empty_hit = int((~has).sum()), int(st["any_pred"][~has].sum())
        point = _values(st, np.arange(len(frames)))
        boots = [_values(st, idx) for idx in boot["index"]]
        ci = {k: _percentiles([b[k] for b in boots]) if point[k] is not None else None for k in point}
        n_all = len(frames)
        ok = n_all >= RANK_FRAMES and n_true >= RANK_OBJECTS
        out[c] = {
            "count_error_mean": _fig(point["count_error_mean"], n_frames, n_true, ci["count_error_mean"]),
            "count_error_mean_abs": _fig(point["count_error_mean_abs"], n_frames, n_true, ci["count_error_mean_abs"]),
            "within_20": _fig(within / n_frames if n_frames else None, n_frames, n_true, wilson(within, n_frames)),
            "within_20_strict": _fig(strict / n_frames if n_frames else None, n_frames, n_true,
                                     wilson(strict, n_frames)),
            "passed_by_one_rule": _fig(within - strict, n_frames, n_true),
            "empty_with_pred": _fig(empty_hit / empty_n if empty_n else None, empty_n, None,
                                    wilson(empty_hit, empty_n)),
            "precision": _fig(point["precision"], n_all, n_true, ci["precision"]),
            "recall": _fig(point["recall"], n_all, n_true, ci["recall"]),
            "ap50": _fig(point["ap50"], n_all, n_true, ci["ap50"]),
            "small_recall": _fig(point["small_recall"], n_all, small_true, ci["small_recall"]),
            "largest_pct_error": _fig(point["largest_pct_error"], n_frames, n_true, ci["largest_pct_error"]),
            "rank_floor": {"ok": ok, "why": "" if ok else
                           f"{n_all} test frames and {n_true} true objects of this class; ranking two models "
                           f"needs at least {RANK_FRAMES} frames and {RANK_OBJECTS} objects"},
        }
    return {"classes": {str(c): v for c, v in out.items()}, "threshold": threshold, "small_px": SMALL_PX,
            "frames": len(frames),
            "intervals": {"level": 0.95, "wilson": list(WILSON),
                          **{k: v for k, v in boot.items() if k != "index"}}}
