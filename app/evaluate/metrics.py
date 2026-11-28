"""The figures of one eval (S9 bite 2a, D91): from the test frames and predictions.jsonl, per class. Every figure is
{"value", "frames", "objects"}: the frames and true objects behind it. Intervals and the rank floor are bite 2b (D92).

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
"""

from __future__ import annotations

import numpy as np

SMALL_PX = 32        # D91: 4 cells of YOLOX's finest stride (8 px)
IOU = 0.5


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


def ap50(scored: list[tuple[float, bool]], n_true: int) -> float | None:
    """COCO's AP at one IoU: precision made non-increasing, read at 101 recall points."""
    if not n_true:
        return None
    scored = sorted(scored, key=lambda s: -s[0])
    tp = np.cumsum([h for _, h in scored]).astype(float)
    fp = np.cumsum([not h for _, h in scored]).astype(float)
    rec, prec = tp / n_true, tp / np.maximum(tp + fp, 1e-12)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    points = np.linspace(0, 1, 101)
    at = np.searchsorted(rec, points, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in at]))


def _fig(value, frames, objects=None) -> dict:
    out = {"value": None if value is None else round(float(value), 6), "frames": int(frames)}
    if objects is not None:
        out["objects"] = int(objects)
    return out


def figures(frames: list[dict], predictions: dict[str, list], class_ids: list[int], factor: float | None,
            threshold: float) -> dict:
    """frames: the test samples ({"file", "width", "height", "boxes": [[class id, x, y, w, h]]}); predictions:
    file -> [[class id, x, y, w, h, score]] (native pixels). Returns {"classes": {id: {figure: {...}}}}."""
    out = {}
    for c in class_ids:
        errs, within, strict, empty_n, empty_hit, scored = [], 0, 0, 0, 0, []
        tp = n_pred = n_true = small_true = small_hit = 0
        largest = []
        for f in frames:
            t = np.array([b[1:5] for b in f["boxes"] if b[0] == c], float).reshape(-1, 4)
            p_all = np.array([b[1:6] for b in predictions.get(f["file"], []) if b[0] == c], float).reshape(-1, 5)
            p = p_all[p_all[:, 4] >= threshold]
            hit_all, _ = match(p_all, t)
            scored += list(zip(p_all[:, 4], hit_all))
            hit, found = match(p, t)
            n_t = len(t)
            tp, n_pred, n_true = tp + int(hit.sum()), n_pred + len(p), n_true + n_t
            if n_t:
                e = len(p) - n_t
                errs.append(e)
                within += abs(e) <= max(1, 0.2 * n_t)
                strict += abs(e) <= 0.2 * n_t
                area = f["width"] * f["height"]
                pred_share = 100 * (p[:, 2] * p[:, 3]).max() / area if len(p) else 0.0
                largest.append(pred_share - 100 * (t[:, 2] * t[:, 3]).max() / area)
                small = np.minimum(t[:, 2], t[:, 3]) * (factor or 1.0) < SMALL_PX
                small_true += int(small.sum())
                small_hit += int((found & small).sum())
            else:
                empty_n += 1
                empty_hit += bool(len(p))
        n_frames = len(errs)
        out[c] = {
            "count_error_mean": _fig(np.mean(errs) if errs else None, n_frames, n_true),
            "count_error_mean_abs": _fig(np.mean(np.abs(errs)) if errs else None, n_frames, n_true),
            "within_20": _fig(within / n_frames if n_frames else None, n_frames, n_true),
            "within_20_strict": _fig(strict / n_frames if n_frames else None, n_frames, n_true),
            "passed_by_one_rule": _fig(within - strict, n_frames, n_true),
            "empty_with_pred": _fig(empty_hit / empty_n if empty_n else None, empty_n),
            "precision": _fig(tp / n_pred if n_pred else None, len(frames), n_true),
            "recall": _fig(tp / n_true if n_true else None, len(frames), n_true),
            "ap50": _fig(ap50(scored, n_true), len(frames), n_true),
            "small_recall": _fig(small_hit / small_true if small_true else None, len(frames), small_true),
            "largest_pct_error": _fig(np.mean(np.abs(largest)) if largest else None, n_frames, n_true),
        }
    return {"classes": {str(c): v for c, v in out.items()}, "threshold": threshold, "small_px": SMALL_PX,
            "frames": len(frames)}
