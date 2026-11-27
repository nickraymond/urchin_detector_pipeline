"""Splits of a dataset version (spec stage 7, DESIGN D61): one fixed test set.

Version 1 takes the export's groups and splits as they are. Later versions never ask the export
where a frame goes (it may deal its groups again on every Export, D43):

1. a frame that was in the version before keeps its split;
2. a group that was in the version before keeps its split, and its new frames join it;
3. a group that is new by name but holds frames of the version before (the export's grouping
   changed from video to site + dive date) takes the most held-out split those frames are in
   (test, then valid, then train); when they sit in more than one split it "straddles" and
   the page lists it. A test frame never becomes a training frame;
4. groups with no earlier frame are dealt, largest first, to the split furthest below
   70 / 20 / 10, counting the frames already placed (the export's rule, from carried counts).
"""

from __future__ import annotations

from app.export.splits import SHARES, SPLITS, site_date

HELD_OUT = ("test", "valid", "train")     # most held-out first


def group_of(source: str, file: str, grouped_by: str) -> str:
    """The export's group for a frame (app/export/splits.split): site + dive date, else the
    source video; by video when the export fell back to it. A still without a source is its
    own group."""
    if grouped_by == "video":
        return source or file
    return site_date(source) or source or file


def assign(frames: list[dict], prev: dict | None) -> tuple[dict[str, str], list[dict]]:
    """frames: [{"file", "group", "split" (the export's)}]; prev: the manifest of the version
    before, or None. Returns ({file: split}, groups) with groups = [{"group", "split",
    "frames", "carried": "export" | "v<N>" | "new", "straddles": [splits]}], by group name."""
    members: dict[str, list[dict]] = {}
    for f in frames:
        members.setdefault(f["group"], []).append(f)
    if prev is None:
        rows = []
        for g in sorted(members):
            seen = sorted({f["split"] for f in members[g]}, key=SPLITS.index)
            rows.append({"group": g, "split": seen[0], "frames": len(members[g]),
                         "carried": "export", "straddles": seen if len(seen) > 1 else []})
        return {f["file"]: f["split"] for f in frames}, rows
    tag = f"v{prev['number']}"
    old_frame = {f["file"]: f["split"] for f in prev["frames"]}
    old_group = {g["group"]: g["split"] for g in prev["groups"]}
    where: dict[str, dict] = {}
    for g, fs in members.items():
        seen = sorted({old_frame[f["file"]] for f in fs if f["file"] in old_frame},
                      key=HELD_OUT.index)
        if g in old_group:
            where[g] = {"split": old_group[g], "carried": tag,
                        "straddles": seen if len(seen) > 1 else []}
        elif seen:
            where[g] = {"split": seen[0], "carried": tag,
                        "straddles": seen if len(seen) > 1 else []}
    by_frame = {f["file"]: old_frame.get(f["file"]) or where[f["group"]]["split"]
                for f in frames if f["group"] in where}
    have = {s: sum(1 for v in by_frame.values() if v == s) for s in SPLITS}
    new = [g for g in members if g not in where]
    for g in sorted(new, key=lambda g: (-len(members[g]), g)):
        s = max(SPLITS, key=lambda s: (SHARES[s] * len(frames) - have[s], -SPLITS.index(s)))
        where[g] = {"split": s, "carried": "new", "straddles": []}
        have[s] += len(members[g])
        by_frame.update({f["file"]: s for f in members[g]})
    return by_frame, [{"group": g, "frames": len(members[g]), **where[g]} for g in sorted(members)]
