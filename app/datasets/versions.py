"""Dataset versions (spec stage 7; DESIGN §Planned for S8, D60, D61).

A version is a frozen snapshot of the project's export, so a model is trained on a version and
never on "whatever the export is today":

    <project>/datasets/images/<sha256>.<ext>   every frame any version uses, stored once
    <project>/datasets/v<N>/manifest.json      frames (by content hash), groups, splits, counts, hash
    <project>/datasets/v<N>/detector_coco.json the export's file, `split` set to this version's
    <project>/datasets/v<N>/frame_metrics.csv  the export's file, `split` set to this version's
    <project>/datasets/v<N>/seg_masks/         the export's PNGs and classes.json, copied
    <project>/datasets/v<N>/splits/            train.txt, valid.txt, test.txt
    <project>/datasets/v<N>/outside/<key>.jsonl  rows of an added public source (bite 3); its
                                               images stay in the outside-data folder

Reads only `export/` (never app.review or the project's database). A version is built in
`v<N>.new/` and renamed when complete; nothing here rewrites or removes a `v<N>/` folder or a
file in `datasets/images/`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.datasets import outside as ds_outside
from app.datasets import splits as ds_splits
from app.export.splits import SPLITS

VERSION_DIR = re.compile(r"^v(\d+)$")
NAME_MAX = 80


class FreezeError(RuntimeError):
    """Freeze cannot run; the message says why and what to do. Nothing was written."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def root(folder: Path) -> Path:
    return Path(folder) / "datasets"


def list_versions(folder: Path) -> list[dict]:
    """Every version's manifest, oldest first. A folder without a readable manifest is not a
    version (a `.new` build never matches the name rule)."""
    found = []
    top = root(folder)
    for p in top.iterdir() if top.is_dir() else []:
        m = VERSION_DIR.match(p.name)
        if m and (p / "manifest.json").is_file():
            try:
                found.append(json.loads((p / "manifest.json").read_text()))
            except (ValueError, OSError):
                continue
    return sorted(found, key=lambda v: v["number"])


def export_ready(folder: Path) -> dict:
    """The export's summary.json, or FreezeError saying what to do first."""
    out = Path(folder) / "export"
    try:
        summary = json.loads((out / "summary.json").read_text())
    except (OSError, ValueError):
        raise FreezeError("No export yet: press Export on the Export page first.") from None
    if not summary.get("ok"):
        raise FreezeError("The last export did not pass its checks: fix it on the Export page "
                          "and export again.")
    return summary


def content_hash(manifest: dict) -> str:
    """The version's identity: classes, label files, frames, groups and outside sources. Not
    its number, name, date or where a group's split was carried from, so the same content
    always gives the same hash."""
    body = {"classes": manifest["classes"], "files": manifest["files"],
            "frames": manifest["frames"], "outside": manifest["outside"],
            "groups": [{k: g[k] for k in ("group", "split", "frames")} for g in manifest["groups"]]}
    return _sha(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def _counts(coco: dict, by_frame: dict[str, str], class_names: dict[int, str]) -> dict:
    """{"frames": {split: n}, "classes": [{id, name, objects: {split: n}, frames: {split: n}}]}:
    frames = every frame of the version; objects and frames per class = detector data (COCO,
    so without the frames the split check left out)."""
    img_split = {i["id"]: by_frame[i["file_name"]] for i in coco["images"]}
    objects: dict[tuple[int, str], int] = {}
    with_class: dict[tuple[int, str], set] = {}
    for a in coco["annotations"]:
        key = (a["category_id"], img_split[a["image_id"]])
        objects[key] = objects.get(key, 0) + 1
        with_class.setdefault(key, set()).add(a["image_id"])
    return {"frames": {s: sum(1 for v in by_frame.values() if v == s) for s in SPLITS},
            "detector_frames": {s: sum(1 for v in img_split.values() if v == s) for s in SPLITS},
            "classes": [{"id": c["id"], "name": class_names.get(c["id"], c["name"]),
                         "objects": {s: objects.get((c["id"], s), 0) for s in SPLITS},
                         "frames": {s: len(with_class.get((c["id"], s), ())) for s in SPLITS}}
                        for c in coco["categories"]]}


def plan(folder: Path, name: str = "", outside_keys=(), outside_dir: Path | None = None) -> dict:
    """Read export/ (and the outside sources asked for) and work out the next version without
    writing anything. Returns {"manifest", "coco" (bytes), "metrics" (bytes), "outside":
    {rows file name: bytes}, "same_as": N | None}."""
    folder = Path(folder)
    summary = export_ready(folder)
    out = folder / "export"
    name = " ".join((name or "").split())
    if len(name) > NAME_MAX:
        raise FreezeError(f"The name is {len(name)} characters; at most {NAME_MAX}.")
    coco = json.loads((out / "detector_coco.json").read_text())
    classes = json.loads((out / "seg_masks" / "classes.json").read_text())["classes"]
    split_info = json.loads((out / "splits" / "splits.json").read_text())
    with (out / "frame_metrics.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        head, rows = list(reader.fieldnames or []), list(reader)
    if not rows:
        raise FreezeError("The export has no frames: accept frames, export, then freeze.")
    size = {i["file_name"]: (i["width"], i["height"]) for i in coco["images"]}
    frames = []
    for r in rows:
        img = out / "images" / r["frame"]
        if r["frame"] not in size:        # left out of COCO by the split check: read its size
            from PIL import Image
            with Image.open(img) as im:
                size[r["frame"]] = im.size
        frames.append({
            "file": r["frame"], "image": sha256_file(img) + img.suffix.lower(),
            "width": size[r["frame"]][0], "height": size[r["frame"]][1], "source": r["source"],
            "group": ds_splits.group_of(r["source"], r["frame"], split_info["grouped_by"]),
            "split": r["split"], "split_ok": r["split_ok"], "seg_mask": r["seg_mask"],
            "checked_classes": [int(i) for i in r["checked_classes"].split()]})
    added, outside_files = [], {}
    for key in sorted(set(outside_keys)):
        if outside_dir is None:
            raise FreezeError("No outside data folder is set: add outside_data_dir to "
                              "config.local.toml.")
        try:
            entry, data = ds_outside.snapshot(outside_dir, key)
        except ds_outside.OutsideError as e:
            raise FreezeError(str(e)) from None
        added.append(entry)
        outside_files[entry["rows"]] = data
    versions = list_versions(folder)
    prev = versions[-1] if versions else None
    by_frame, groups = ds_splits.assign(frames, versions)
    for f in frames:
        f["split"] = by_frame[f["file"]]
    for i in coco["images"]:
        i["split"] = by_frame[i["file_name"]]
    for r in rows:
        r["split"] = by_frame[r["frame"]]
    coco_bytes = json.dumps(coco, separators=(",", ":")).encode()
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=head)
    w.writeheader()
    w.writerows(rows)
    metrics_bytes = buf.getvalue().encode()
    masks = sorted({f["seg_mask"] for f in frames})
    manifest = {
        "number": (prev["number"] + 1) if prev else 1, "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "previous": prev["number"] if prev else None,
        "export": {"at": summary.get("at"), "frames": summary.get("frames"),
                   "objects": summary.get("objects"), "split_no": summary.get("split_no")},
        "grouped_by": split_info["grouped_by"], "split_warning": split_info.get("warning"),
        "classes": classes,
        "files": {"detector_coco.json": _sha(coco_bytes), "frame_metrics.csv": _sha(metrics_bytes),
                  "seg_masks/classes.json": sha256_file(out / "seg_masks" / "classes.json"),
                  **{f"seg_masks/{m}": sha256_file(out / "seg_masks" / m) for m in masks}},
        "frames": frames, "groups": groups, "outside": added,
        "counts": _counts(coco, by_frame, {c["id"]: c["name"] for c in classes})}
    manifest["hash"] = content_hash(manifest)
    same = next((v["number"] for v in reversed(versions) if v.get("hash") == manifest["hash"]), None)
    return {"manifest": manifest, "coco": coco_bytes, "metrics": metrics_bytes,
            "outside": outside_files, "same_as": same}


def _copy_once(src: Path, dst: Path) -> None:
    """Copy under a temporary name, then rename: a stopped copy never leaves half a file. A file
    already there with the same size is trusted here (its name is its content hash); the loader
    checks the hash itself before a frame is first used (loaders.verify_image)."""
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def freeze(folder: Path, name: str = "",
           progress: Callable[[int, int, str], None] | None = None,
           outside_keys=(), outside_dir: Path | None = None) -> dict:
    """Freeze the current export as the next version. Returns {"number", "new": bool,
    "manifest"}; new is False when nothing changed since an earlier version (none is written)."""
    folder = Path(folder)
    p = plan(folder, name, outside_keys, outside_dir)
    m = p["manifest"]
    if p["same_as"] is not None:
        return {"number": p["same_as"], "new": False, "manifest": m}
    out, top = folder / "export", root(folder)
    (top / "images").mkdir(parents=True, exist_ok=True)
    for stale in top.glob("v*.new"):
        shutil.rmtree(stale)
    new, dest = top / f"v{m['number']}.new", top / f"v{m['number']}"
    if dest.exists():
        raise FreezeError(f"{dest} already exists without a manifest: move it aside first.")
    (new / "seg_masks").mkdir(parents=True)
    (new / "splits").mkdir()
    for n, f in enumerate(m["frames"], 1):
        src = out / "images" / f["file"]
        if sha256_file(src) + src.suffix.lower() != f["image"]:
            raise FreezeError(f"{f['file']} changed in export/ while freezing: freeze again.")
        _copy_once(src, top / "images" / f["image"])
        shutil.copyfile(out / "seg_masks" / f["seg_mask"], new / "seg_masks" / f["seg_mask"])
        if progress:
            progress(n, len(m["frames"]), f["file"])
    shutil.copyfile(out / "seg_masks" / "classes.json", new / "seg_masks" / "classes.json")
    if p["outside"]:
        (new / "outside").mkdir()
        for rows_name, data in p["outside"].items():
            (new / "outside" / rows_name).write_bytes(data)
    (new / "detector_coco.json").write_bytes(p["coco"])
    (new / "frame_metrics.csv").write_bytes(p["metrics"])
    for s in SPLITS:
        (new / "splits" / f"{s}.txt").write_text(
            "".join(f"{f['file']}\n" for f in m["frames"] if f["split"] == s))
    for rel, want in m["files"].items():
        if sha256_file(new / rel) != want:
            raise FreezeError(f"{rel} changed in export/ while freezing: freeze again.")
    (new / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    os.replace(new, dest)
    return {"number": m["number"], "new": True, "manifest": m}
