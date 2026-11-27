"""Outside data for detector pre-training (S8 bite 3; spec stage 7, Nick 2026-11-26 / 27).

Public urchin datasets from Nick's earlier work live outside git in one folder (the
`outside_data_dir` setting; on his laptop `~/nereus_ml/datasets`). Their licence captures and
verdicts are in the repo ADIN_SPI_OpenMV: `ml/urchin_data/licenses/`, `ml/urchin_data/manifests/`
and `docs/urchin_datasets.md` §Verification (read 2026-11-27 at `994b8c2`). A dataset version
may add a source only when it is listed in SOURCES with `allowed: True`, i.e. its recorded
verdict allows commercial use and no licence question is open on it; every other folder found
there is refused, with the reason shown. Today that is Urchinbot only (D63: RF100 waits for
Nick); it is the default outside source, ticked on the page when its folder is there.

Outside boxes are one class, "urchin": they carry no Build 1 class id. They are used for
pre-training the detector only, in the train and valid splits; never in the test split, which
is always Nick's own reviewed frames. A source's own test images are not used at all.

Each source is read through the `labels.jsonl` its folder holds (earlier work's convention,
`ml/yolox_urchin/data.py`): one line per image, {"file", "w", "h", "classes", "boxes":
[[class index, x0, y0, w, h, pixels]]}, absolute pixels. Images are referenced in place (the
folder is tens of GB and its archives are never modified); the version records each image's
size in bytes and the loader checks it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CLASS_NAME = "urchin"
USED_SPLITS = ("train", "valid")

SOURCES = [
    {"key": "urchinbot", "name": "Urchinbot (U. Auckland)", "folder": "urchinbot",
     "allowed": True, "licence": "CC-BY-4.0",
     "licence_source": "Zenodo record 16060266 (DOI 10.5281/zenodo.16060266), licence field "
                       "cc-by-4.0; captured 2026-08-21 in ADIN_SPI_OpenMV "
                       "ml/urchin_data/licenses/urchinbot.md",
     "verdict": "GO for backbone (docs/urchin_datasets.md §Verification, 2026-08-21)",
     "attribution": "Urchinbot Dataset - Sea Urchin Object Detection and Classification, "
                    "Zenodo, https://doi.org/10.5281/zenodo.16060266, CC BY 4.0",
     "recorded": {"images": 9872, "boxes": 44268},
     # the source's own lists; an image in two lists goes to the more held-out one
     "split_lists": {"train": "archives/train.txt", "valid": "archives/val.txt",
                     "test": "archives/test.txt"},
     "note": "Its companion GitHub weights and code have no licence and are not used."},
    {"key": "rf100_underwater", "name": "RF100 underwater-objects (Roboflow 100)",
     "folder": "roboflow/rf100_underwater", "allowed": False, "licence": "CC BY 4.0",
     "why_not": "Waits for Nick: same image lineage as DUO (URPC contest images). Its own "
                "record says CC BY 4.0 and GO, but a licence question is Nick's (Lead call D63).",
     "licence_source": "the export's data.yaml (roboflow: license: CC BY 4.0) and the Universe "
                       "page; captured 2026-08-21 in ADIN_SPI_OpenMV "
                       "ml/urchin_data/licenses/roboflow_tier1b.md",
     "verdict": "GO (docs/urchin_datasets.md §Verification, 2026-08-21)",
     "attribution": "underwater-objects-5v7p8 v2, Roboflow 100, "
                    "https://universe.roboflow.com/roboflow-100/underwater-objects-5v7p8, "
                    "CC BY 4.0",
     "recorded": {"images": 7600, "boxes": 25299},
     "split_prefix": {"train": "train/", "valid": "valid/", "test": "test/"},
     },
    {"key": "urchin_detector_74", "name": "Roboflow sea-urchin/urchin-detector (74 images)",
     "folder": "roboflow/urchin_detector_74", "allowed": False, "licence": "CC BY 4.0",
     "why_not": "Recorded as hard-case evaluation material, not pre-training data; and the "
                "label file on disk (2,078 boxes, 2 classes) does not match the verified "
                "export record (2,109 boxes, 4 classes)."},
    {"key": "duo", "name": "DUO", "folder": "duo", "allowed": False,
     "licence": "figshare declares CC BY 4.0; contest lineage unclear",
     "why_not": "Nick, 2026-11-27: the DUO set stays out, and so do weights trained on it."},
    {"key": "fathomnet", "name": "FathomNet S. fragilis", "folder": "fathomnet",
     "allowed": False, "licence": "CC BY-NC-ND (all 4,600 images)",
     "why_not": "No commercial use (recorded verdict: NO-GO commercial)."},
    {"key": "noaa_hf_weights", "name": "NOAA yolo11 urchin weights", "folder": "noaa_hf_weights",
     "allowed": False, "licence": "AGPL-3.0 lineage (Ultralytics)",
     "why_not": "Benchmark-only in the record; Ultralytics models are not allowed in the app."},
    {"key": "gbif_inat", "name": "iNaturalist via GBIF (CC0 / CC-BY subset)",
     "folder": "gbif_inat", "allowed": False, "licence": "CC0-1.0 and CC-BY-4.0 per record",
     "why_not": "Species labels per image, no reviewed boxes: not detector data."},
    {"key": "corpus_v1", "name": "corpus_v1 (earlier training corpus)", "folder": "corpus_v1",
     "allowed": False, "licence": "mixed", "why_not": "Built from Urchinbot + DUO + RF100 "
     "(its manifest.json): contains the DUO set."},
    {"key": "corpus_v2", "name": "corpus_v2 (earlier training corpus)", "folder": "corpus_v2",
     "allowed": False, "licence": "mixed", "why_not": "Its manifest.json lists DUO among its "
     "sources."},
]
BY_KEY = {s["key"]: s for s in SOURCES}


class OutsideError(RuntimeError):
    """A source cannot be added; the message says why."""


def _split_of(src: dict, root: Path) -> callable:
    """file (as in labels.jsonl) -> "train" | "valid" | "test" | None (in no list)."""
    if "split_prefix" in src:
        pre = src["split_prefix"]
        return lambda f: next((s for s, p in pre.items() if f.startswith(p)), None)
    names: dict[str, str] = {}
    for s in ("train", "valid", "test"):          # later lists win: test over valid over train
        for line in (root / src["split_lists"][s]).read_text().splitlines():
            if line.strip():
                names[Path(line.strip()).name] = s
    return lambda f: names.get(Path(f).name)


def read(outside_dir: Path, key: str) -> dict:
    """Read one allowed source. Returns {"rows": [{file, w, h, split, boxes: [[x, y, w, h]]}]
    (train and valid only), "counts": {split: {images, boxes}}, "unused": {"test": n,
    "no_split": n}, "on_disk": {images, boxes}}. Boxes of every class in the file count as
    "urchin" only when the file's class is urchin-like; other classes raise (the earlier
    normalisation already keeps urchin boxes only)."""
    src = BY_KEY.get(key)
    if src is None or not src["allowed"]:
        raise OutsideError(f"{key!r} is not an allowed outside source"
                           + (f": {src['why_not']}" if src else "."))
    root = Path(outside_dir) / src["folder"]
    labels = root / "labels.jsonl"
    if not labels.is_file():
        raise OutsideError(f"{src['name']}: {labels} not found. Check outside_data_dir in "
                           "config.local.toml.")
    split_of = _split_of(src, root)
    rows, unused = [], {"test": 0, "no_split": 0}
    counts = {s: {"images": 0, "boxes": 0} for s in USED_SPLITS}
    on_disk = {"images": 0, "boxes": 0}
    with labels.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            bad = sorted({r["classes"][b[0]] for b in r["boxes"]} - {CLASS_NAME})
            if bad:
                raise OutsideError(f"{src['name']}: {r['file']} has boxes of class {bad}; only "
                                   f"{CLASS_NAME!r} is read.")
            on_disk["images"] += 1
            on_disk["boxes"] += len(r["boxes"])
            split = split_of(r["file"])
            if split not in USED_SPLITS:
                unused["test" if split == "test" else "no_split"] += 1
                continue
            rows.append({"file": r["file"], "w": r["w"], "h": r["h"], "split": split,
                         "boxes": [[b[1], b[2], b[3], b[4]] for b in r["boxes"]]})
            counts[split]["images"] += 1
            counts[split]["boxes"] += len(r["boxes"])
    return {"rows": rows, "counts": counts, "unused": unused, "on_disk": on_disk}


def catalogue(outside_dir: Path | None) -> list[dict]:
    """Every known source as the Datasets page lists it: licence, verdict, whether its folder
    is there, and whether it can be added. Folders in outside_dir that no record names are
    listed as not allowed (no licence record). Reads no label file (the page stays fast)."""
    out = []
    root = Path(outside_dir) if outside_dir else None
    for s in SOURCES:
        present = bool(root and (root / s["folder"]).is_dir())
        out.append({**s, "present": present,
                    "can_add": bool(s["allowed"] and present
                                    and (root / s["folder"] / "labels.jsonl").is_file())})
    known = {s["folder"].split("/")[0] for s in SOURCES}
    for p in sorted(root.iterdir()) if root and root.is_dir() else []:
        if p.is_dir() and p.name not in known and not p.name.startswith("."):
            out.append({"key": p.name, "name": p.name, "folder": p.name, "allowed": False,
                        "licence": "no record", "present": True, "can_add": False,
                        "why_not": "No licence record for this folder: not used."})
    return out


def snapshot(outside_dir: Path, key: str) -> tuple[dict, bytes]:
    """A source as a dataset version freezes it: (the manifest entry, the rows file's bytes).
    The rows file is a .jsonl, one row per image with the image's size in bytes; a version
    stores it as `outside/<key>.jsonl` and the entry carries its sha256."""
    got = read(outside_dir, key)
    src = BY_KEY[key]
    root = Path(outside_dir) / src["folder"]
    lines = []
    for r in got["rows"]:
        p = root / r["file"]
        if not p.is_file():
            raise OutsideError(f"{src['name']}: image {p} is missing.")
        lines.append(json.dumps({**r, "bytes": p.stat().st_size}, separators=(",", ":")) + "\n")
    data = "".join(lines).encode()
    return {"key": key, "name": src["name"], "licence": src["licence"],
            "licence_source": src["licence_source"], "attribution": src["attribution"],
            "class": CLASS_NAME, "use": "detector pre-training only; never the test split",
            "folder": src["folder"], "rows": f"{key}.jsonl",
            "sha256": hashlib.sha256(data).hexdigest(), "counts": got["counts"],
            "unused": got["unused"], "recorded": src["recorded"],
            "on_disk": got["on_disk"]}, data


def samples(outside_dir: Path, version_dir: Path, entry: dict, split: str) -> list[dict]:
    """Pre-training samples of one source of a version: [{"file", "image" (path), "bytes",
    "width", "height", "boxes": [[x, y, w, h]]}] (one class). The test split has none."""
    if split not in USED_SPLITS:
        return []
    root = Path(outside_dir) / entry["folder"]
    out = []
    with (Path(version_dir) / "outside" / entry["rows"]).open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == split:
                out.append({"file": r["file"], "image": str(root / r["file"]),
                            "bytes": r["bytes"], "width": r["w"], "height": r["h"],
                            "boxes": r["boxes"]})
    return out
