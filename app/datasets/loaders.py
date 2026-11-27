"""What a detector trainer gets from a dataset version (S8 bite 2b; spec "What gets trained").

Reads only `datasets/v<N>/` and `datasets/images/`; no torch here, so the rules are tested
without it. The rules, from Build 1's review flags:

- a frame whose split check is "no" is not detector data (it is not in the version's COCO);
- a class not marked fully checked on a frame is skipped for that frame: its boxes are not
  targets and the frame is never used as "no <class> here". Its labeled boxes come back as
  `ignore` regions, so a trainer can keep them out of the background too;
- a frame on which none of the run's classes is fully checked teaches nothing: left out;
- classes are the export's ids; which ones a model learns is a choice per run. A box of a class
  the run does not train is not a target and not an ignore region (the model is not asked
  about it).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.datasets import versions

# Build 2 trains on sea stars and urchins only (Nick, 2026-11-27; SPEC Q31). Still a setting
# per run: these are only the classes ticked by default, matched by name in the version.
DEFAULT_CLASS_NAMES = ("purple urchin", "red urchin", "black urchin", "sea star",
                       "sunflower sea star")


class DatasetError(RuntimeError):
    """The version cannot be used as asked; the message says why."""


def load(folder: Path, number: int) -> tuple[dict, dict]:
    """(manifest, coco) of version `number`."""
    vdir = versions.root(folder) / f"v{int(number)}"
    try:
        manifest = json.loads((vdir / "manifest.json").read_text())
        coco = json.loads((vdir / "detector_coco.json").read_text())
    except (OSError, ValueError) as e:
        raise DatasetError(f"Dataset version {number} cannot be read ({e}).") from None
    return manifest, coco


def object_classes(manifest: dict) -> list[dict]:
    return [c for c in manifest["classes"] if c["kind"] == "object"]


def default_class_ids(manifest: dict) -> list[int]:
    return [c["id"] for c in object_classes(manifest) if c["name"] in DEFAULT_CLASS_NAMES]


def check_classes(manifest: dict, class_ids) -> list[int]:
    known = {c["id"] for c in object_classes(manifest)}
    ids = sorted({int(i) for i in class_ids})
    bad = [i for i in ids if i not in known]
    if bad or not ids:
        raise DatasetError(f"Class id(s) {bad} are not object classes of version "
                           f"{manifest['number']}." if bad else "Pick at least one class.")
    return ids


def detector_samples(folder: Path, number: int, split: str, class_ids=None) -> dict:
    """{"samples": [...], "left_out": [{"file", "why"}], "class_ids": [...]} for one split.
    A sample: {"file", "image" (path in datasets/images/), "sha256", "width", "height",
    "boxes": [[class_id, x, y, w, h]] (targets), "checked": [class ids that are fully checked
    here, of the run's], "ignore": [[x, y, w, h]] (boxes of the run's classes not fully
    checked here)}. Boxes are COCO style, absolute pixels, x / y = top-left."""
    manifest, coco = load(folder, number)
    ids = check_classes(manifest, class_ids if class_ids is not None
                        else default_class_ids(manifest))
    by_image: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    coco_img = {i["file_name"]: i for i in coco["images"]}
    samples, left_out = [], []
    for f in manifest["frames"]:
        if f["split"] != split:
            continue
        img = coco_img.get(f["file"])
        if f["split_ok"] == "no" or img is None:
            left_out.append({"file": f["file"], "why": "split check: no (cover only)"})
            continue
        checked = [i for i in ids if i in f["checked_classes"]]
        if not checked:
            left_out.append({"file": f["file"], "why": "none of the run's classes is fully "
                                                       "checked on it"})
            continue
        boxes, ignore = [], []
        for a in by_image.get(img["id"], []):
            if a["category_id"] in checked:
                boxes.append([a["category_id"], *a["bbox"]])
            elif a["category_id"] in ids:
                ignore.append(list(a["bbox"]))
        sha, _, ext = f["image"].partition(".")
        samples.append({"file": f["file"], "sha256": sha, "width": f["width"],
                        "height": f["height"], "boxes": boxes, "checked": checked,
                        "ignore": ignore,
                        "image": str(versions.root(folder) / "images" / f["image"])})
    return {"samples": samples, "left_out": left_out, "class_ids": ids}


def run_frames(samples: list[dict], class_ids: list[int]) -> tuple[list[dict], list[dict]]:
    """(fully checked frames, boxes-only frames) of a detector run (D66). The trainer, the
    Train page's check and the Datasets page's "what training will see" all call this.

    - fully checked: EVERY class of the run is fully checked on the frame. It trains as usual
      (also "nothing here" where nothing is labeled) and it is the only kind used for validation.
    - boxes-only: at least one class of the run is fully checked, at least one is not. An
      animal of the unchecked class may sit there unlabeled, and a detector's "is there an
      object" output has no class, so nowhere on such a frame may the model be taught "nothing
      here" (spec: "never treated as background"). It trains on its labeled boxes only.
    Frames with none of the run's classes checked never get here (detector_samples)."""
    want = sorted(class_ids)
    full = [s for s in samples if sorted(s["checked"]) == want]
    return full, [s for s in samples if sorted(s["checked"]) != want]


def usable(samples: list[dict], class_ids: list[int]) -> tuple[list[dict], int]:
    """(fully checked frames, how many boxes-only frames): the frames a run validates on."""
    full, boxes_only = run_frames(samples, class_ids)
    return full, len(boxes_only)


def verify_image(sample: dict) -> None:
    """The stored frame must still be the frozen one (trainers call this on first use)."""
    p = Path(sample["image"])
    if not p.is_file():
        raise DatasetError(f"{p} is missing: the dataset's image store was changed by hand.")
    if versions.sha256_file(p) != sample["sha256"]:
        raise DatasetError(f"{p} no longer matches its hash: the dataset's image store was "
                           "changed by hand.")
