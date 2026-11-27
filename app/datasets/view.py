"""What training will see in a dataset version (S8 bite 2b; spec stage 7): per class and split
the objects and frames a detector run would get after the review flags are honored, what is
skipped and why, and which classes are under the 150–200 object target. The numbers come from
the same function the trainer's loader uses (loaders.detector_samples), so the page cannot
promise data the trainer will not get."""

from __future__ import annotations

from pathlib import Path

from app.datasets import loaders
from app.export.splits import SPLITS

TARGET = (150, 200)     # objects per class (docs/source/mvp_spec.md; the Review page's totals)


def training_view(folder: Path, number: int, class_ids=None) -> dict:
    """{"classes": [{id, name, default, seen: {split: {objects, frames}}, skipped: {split:
    {objects, frames}}, total, under_target}], "split_no": [files], "unchecked": [{file, split,
    classes: [names]}], "frames": {split: {used, left_out}}, "target": TARGET}.
    seen.frames = frames where the class is fully checked (also with zero of it: real
    negatives); skipped = the class is labeled or wanted there but not fully checked."""
    manifest, coco = loaders.load(folder, number)
    objs = loaders.object_classes(manifest)
    defaults = loaders.default_class_ids(manifest)
    ids = loaders.check_classes(manifest, class_ids if class_ids is not None else defaults)
    name = {c["id"]: c["name"] for c in objs}
    zero = lambda: {s: {"objects": 0, "frames": 0} for s in SPLITS}   # noqa: E731
    seen, skipped = {i: zero() for i in ids}, {i: zero() for i in ids}
    frames = {}
    labeled: dict[tuple[str, int], int] = {}
    img_file = {i["id"]: i["file_name"] for i in coco["images"]}
    for a in coco["annotations"]:
        key = (img_file[a["image_id"]], a["category_id"])
        labeled[key] = labeled.get(key, 0) + 1
    for s in SPLITS:
        got = loaders.detector_samples(folder, number, s, ids)
        frames[s] = {"used": len(got["samples"]), "left_out": len(got["left_out"])}
        for smp in got["samples"]:
            for i in smp["checked"]:
                seen[i][s]["frames"] += 1
            for b in smp["boxes"]:
                seen[b[0]][s]["objects"] += 1
    unchecked = []
    for f in manifest["frames"]:
        if f["split_ok"] == "no":
            continue
        missing = [i for i in ids if i not in f["checked_classes"]]
        for i in missing:
            skipped[i][f["split"]]["frames"] += 1
            skipped[i][f["split"]]["objects"] += labeled.get((f["file"], i), 0)
        if missing:
            unchecked.append({"file": f["file"], "split": f["split"],
                              "classes": [name[i] for i in missing]})
    rows = []
    for i in ids:
        total = sum(v["objects"] for v in seen[i].values())
        rows.append({"id": i, "name": name[i], "default": i in defaults, "seen": seen[i],
                     "skipped": skipped[i], "total": total,
                     "skipped_total": sum(v["objects"] for v in skipped[i].values()),
                     "under_target": total < TARGET[0]})
    return {"number": manifest["number"], "classes": rows, "frames": frames,
            "split_no": [f["file"] for f in manifest["frames"] if f["split_ok"] == "no"],
            "unchecked": unchecked, "target": TARGET,
            "all_classes": [{"id": c["id"], "name": c["name"], "default": c["id"] in defaults,
                             "picked": c["id"] in ids} for c in objs]}
