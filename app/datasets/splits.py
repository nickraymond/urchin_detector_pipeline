"""Splits of a dataset version (spec stage 7, DESIGN D61): one fixed test set.

Version 1 takes the export's groups and splits as they are. Later versions never ask the export
where a frame goes (it may deal its groups again on every Export, D43):

1. a frame that was in an earlier version keeps the split it had in the newest version that
   holds it (so a frame left out of one version and back in the next is not dealt again);
2. a group that was in an earlier version keeps its split (again from the newest version that
   holds it), and its new frames join it;
3. a group that is new by name but holds earlier frames (the export's grouping changed from
   video to site + dive date) takes the most held-out split those frames are in (test, then
   valid, then train); when they sit in more than one split it "straddles" and the page
   lists it. A test frame never becomes a training frame;
4. groups with no earlier frame are dealt, largest first, to the split furthest below
   70 / 20 / 10, counting the frames already placed (the export's rule, from carried counts).

"The same frame" means the same file name (a frame's name carries its video, path code and
time, and never changes), not the same image bytes: a rotated still is re-encoded on every
Export (D44), so its content hash can change while it is still that frame.
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


def assign(frames: list[dict], earlier: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """frames: [{"file", "group", "split" (the export's)}]; earlier: the manifests of every
    earlier version, oldest first (empty for version 1). Returns ({file: split}, groups) with
    groups = [{"group", "split", "frames", "carried": "export" | "v<N>" | "new",
    "straddles": [splits]}], by group name; v<N> = the newest version the split came from."""
    members: dict[str, list[dict]] = {}
    for f in frames:
        members.setdefault(f["group"], []).append(f)
    if not earlier:
        rows = []
        for g in sorted(members):
            seen = sorted({f["split"] for f in members[g]}, key=SPLITS.index)
            rows.append({"group": g, "split": seen[0], "frames": len(members[g]),
                         "carried": "export", "straddles": seen if len(seen) > 1 else []})
        return {f["file"]: f["split"] for f in frames}, rows
    old_frame: dict[str, tuple[str, int]] = {}       # file -> (split, version); newest wins
    old_group: dict[str, tuple[str, int]] = {}
    for v in earlier:
        old_frame.update({f["file"]: (f["split"], v["number"]) for f in v["frames"]})
        old_group.update({g["group"]: (g["split"], v["number"]) for g in v["groups"]})
    where: dict[str, dict] = {}
    for g, fs in members.items():
        held = [old_frame[f["file"]] for f in fs if f["file"] in old_frame]
        seen = sorted({s for s, _ in held}, key=HELD_OUT.index)
        if g in old_group or seen:
            # the group's own earlier split and the splits its earlier frames sit in: the most
            # held-out wins. They differ when the grouping changed in between (video -> dive ->
            # video, QE PR 63): a video whose dive was held out meanwhile must not get new
            # training frames next to that dive's test frames.
            found = sorted(set(seen) | ({old_group[g][0]} if g in old_group else set()),
                           key=HELD_OUT.index)
            newest = max([n for _, n in held] + ([old_group[g][1]] if g in old_group else []))
            where[g] = {"split": found[0], "carried": f"v{newest}",
                        "straddles": found if len(found) > 1 else []}
    by_frame = {f["file"]: old_frame[f["file"]][0] if f["file"] in old_frame
                else where[f["group"]]["split"] for f in frames if f["group"] in where}
    have = {s: sum(1 for v in by_frame.values() if v == s) for s in SPLITS}
    new = [g for g in members if g not in where]
    for g in sorted(new, key=lambda g: (-len(members[g]), g)):
        s = max(SPLITS, key=lambda s: (SHARES[s] * len(frames) - have[s], -SPLITS.index(s)))
        where[g] = {"split": s, "carried": "new", "straddles": []}
        have[s] += len(members[g])
        by_frame.update({f["file"]: s for f in members[g]})
    return by_frame, [{"group": g, "frames": len(members[g]), **where[g]} for g in sorted(members)]
