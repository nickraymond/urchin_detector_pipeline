"""The `evaluate` job (S9 bite 1, D90): one trained model × one test set → models/<id>/eval/<version>-<test hash>/.

    predictions.jsonl   one line per test frame: its boxes [class id, x, y, w, h, score] in native pixels, every
                        score down to SCORE_FLOOR (mAP50 needs them all, D91); appended frame by frame
    eval.json           written last: the model file and its sha256, version, test-set hash, the eval scale and
                        tiling used, the threshold, code version, times; its presence means "evaluated"

The test set is the model's dataset version's test split, frames fully checked for the run's classes (D66). An
eval is keyed by version and the test set's hash and never overwritten. App side only (no torch): what can be
evaluated, the params, progress. The worker is app/evaluate/worker.py."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from app.datasets import loaders
from app.jobctl import JobError, Kind
from app.train import models, tiles

PREDICTORS = {"yolox": "app.evaluate.yolox:Predictor"}      # recipe train.trainer -> its predictor
THRESHOLD = 0.3          # D91: the figures' score threshold (the trainer's validation one); never tuned on test
SCORE_FLOOR = 0.01       # predictions are kept down to this score, for mAP50 over all scores (D91)
NMS_IOU = 0.45           # the trainer's (app/train/yolox.py)


def eval_dir(folder: Path, model_id: str, key: str) -> Path:
    return models.root(folder) / model_id / "eval" / key


def test_frames(folder: Path, version: int, class_ids: list[int]) -> list[dict]:
    """The test split's frames fully checked for every class of the run (D66)."""
    full, _ = loaders.run_frames(loaders.detector_samples(folder, version, "test", class_ids)["samples"], class_ids)
    return full


def test_groups(folder: Path, version: int) -> dict[str, str] | None:
    """file -> the frame's site-and-date group (D43) for D92's bootstrap; None when the version was split by video
    (the export could not read site and date), as those groups are not dives."""
    manifest, _ = loaders.load(folder, version)
    if manifest.get("grouped_by") == "video":
        return None
    return {f["file"]: f["group"] for f in manifest["frames"] if f.get("group")}


def test_hash(frames: list[dict], class_ids: list[int]) -> str:
    body = json.dumps({"classes": sorted(class_ids), "frames": sorted([f["file"], f.get("sha256", "")] for f in frames)})
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def scale_for(folder: Path, train: dict) -> dict:
    """The eval factor, overlap and whether it was capped: the recipe's (stored at the run's first start, S8
    Fixes 2), or for a model trained before that, worked out the same way from its version's training split."""
    if train.get("eval_scale"):
        return {"eval_scale": train["eval_scale"], "eval_overlap": int(train.get("eval_overlap", 0)),
                "eval_overlap_capped": bool(train.get("eval_overlap_capped", False)), "source": "recipe"}
    ids, canvas = list(train["class_ids"]), int(train["tile"])
    full, boxes_only = loaders.run_frames(
        loaders.detector_samples(folder, int(train["dataset"]["version"]), "train", ids)["samples"], ids)
    scale = tiles.eval_scale(full + boxes_only)
    f = scale["factor"]
    return {"eval_scale": scale, "eval_overlap": tiles.overlap(full + boxes_only, f, canvas) if f else 0,
            "eval_overlap_capped": bool(f and tiles.wanted_overlap(full + boxes_only, f) > tiles.overlap(
                full + boxes_only, f, canvas)),
            "source": f"worked out from version {train['dataset']['version']}'s training split (the recipe predates it)"}


def params(folder: Path, model_id: str, predictor: str | None = None) -> dict:
    """What the page submits for one model. Refused (JobError) with the reason when it cannot be evaluated."""
    folder, mdir = Path(folder), models.root(folder) / model_id
    try:
        recipe = models.read_recipe(mdir)
    except models.ModelError as e:
        raise JobError(str(e)) from None
    train = recipe.get("train", {})
    if not (mdir / "checkpoints" / "best.pt").is_file():     # QE PR 111: the Train page's reason
        every = int(train.get("val_every") or 1)
        raise JobError(f"Model {model_id} has no best.pt: it is taken only from validated epochs, and this run "
                       f"stopped before its first one (after epoch {min(every, int(train.get('epochs', every)))}). "
                       "Resume it past that epoch, or train again.")
    if train.get("data") == "outside":
        raise JobError(f"Model {model_id} is pre-trained on outside data (one class, \"urchin\"): it has none of "
                       "the project's classes to score. Evaluate a run started from it instead.")
    if not predictor and train.get("trainer") not in PREDICTORS:
        raise JobError(f"No evaluation is built for trainer {train.get('trainer')!r} yet.")
    version, ids = int(train["dataset"]["version"]), list(train["class_ids"])
    try:
        frames = test_frames(folder, version, ids)
    except loaders.DatasetError as e:
        raise JobError(str(e)) from None
    if not frames:
        raise JobError(f"Version {version}'s test split has no frame with every class of model {model_id} fully "
                       "checked: there is nothing to score (the Datasets page shows what is skipped).")
    key = f"v{version}-{test_hash(frames, ids)}"
    if (eval_dir(folder, model_id, key) / "eval.json").is_file():
        raise JobError(f"Model {model_id} is already evaluated on this test set ({key}); evals are never "
                       "overwritten.")
    return {"folder": str(folder), "model_id": model_id, "version": version, "class_ids": ids, "key": key,
            "frames": len(frames), "predictor": predictor or PREDICTORS[train["trainer"]], **scale_for(folder, train)}


def _lines(folder: Path) -> list[tuple[str, dict]]:
    """predictions.jsonl's whole, readable lines (text, parsed): a line cut by a crash is not one."""
    p = Path(folder) / "predictions.jsonl"
    if not p.is_file():
        return []
    out = []
    for text in p.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            line = json.loads(text)
            if text.endswith("\n") and isinstance(line, dict) and "file" in line:
                out.append((text, line))
        except ValueError:
            continue            # a line cut by a crash: that frame is scored again
    return out


def done_files(folder: Path) -> list[str]:
    return [line["file"] for _, line in _lines(folder)]


def trim(folder: Path) -> int:
    """QE PR 121: on every start, cut predictions.jsonl back to its whole lines, one per frame (the first), so a
    new line never lands on a cut one. Returns how many lines were dropped."""
    p = Path(folder) / "predictions.jsonl"
    if not p.is_file():
        return 0
    kept, seen = [], set()
    for text, line in _lines(folder):
        if line["file"] not in seen:
            seen.add(line["file"])
            kept.append(text)
    dropped = len(p.read_text(encoding="utf-8").splitlines()) - len(kept)
    if dropped:
        tmp = p.with_name(".predictions.jsonl.tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        os.replace(tmp, p)
    return dropped


def read_predictions(folder: Path, files: list[str]) -> dict[str, list]:
    """file -> boxes, refused unless every test frame is in the file exactly once (QE PR 121)."""
    lines = [line for _, line in _lines(folder)]
    got = [line["file"] for line in lines]
    missing, extra = set(files) - set(got), set(got) - set(files)
    twice = {f for f in got if got.count(f) > 1}
    if missing or extra or twice:
        raise RuntimeError(f"predictions.jsonl does not hold every test frame once ({len(missing)} missing, "
                           f"{len(twice)} twice, {len(extra)} not in the test set, e.g. "
                           f"{sorted(missing | twice | extra)[:3]}): no eval.json written. Start the evaluation again.")
    return {line["file"]: line["boxes"] for line in lines}


def progress(p: dict) -> tuple[int, int]:
    return len(set(done_files(eval_dir(p["folder"], p["model_id"], p["key"])))), int(p["frames"])


def evaluate_kind() -> Kind:
    return Kind("evaluate", module="app.evaluate.worker", progress=progress, item="frame")
