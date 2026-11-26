"""Train / valid / test by site and dive date (spec §Outputs, Lead call D43).

Site and dive date come from the source file name, ReefCheck style `<site>-<yy>-<mm>-<dd>-…`
(SPEC facts; a site may itself contain dashes), else from its folder's name, else the group is
the source file itself. Whole groups go to one split, so near-identical frames never leak
between sets. Groups are assigned largest first to the split furthest below its share of
70 / 20 / 10 frames; the same input always gives the same answer. With fewer than 3 groups a
fair split is impossible: frames are grouped by source video instead and a warning says so.
"""

from __future__ import annotations

import re
from pathlib import Path

SPLITS = ("train", "valid", "test")
SHARES = {"train": 0.7, "valid": 0.2, "test": 0.1}
SITE_DATE = re.compile(r"^(?P<site>.+?)-(?P<yy>\d{2})-(?P<mm>\d{2})-(?P<dd>\d{2})(?:-|$)")


def site_date(source: str) -> str | None:
    """'<site> 20yy-mm-dd' from a file name, else from its folder's name, else None."""
    p = Path(source)
    for name in (p.stem, p.parent.name):
        m = SITE_DATE.match(name)
        if m:
            return f"{m['site']} 20{m['yy']}-{m['mm']}-{m['dd']}"
    return None


def assign(groups: dict[str, int]) -> dict[str, str]:
    """{group: split} for {group: frame count}: largest group first, each to the split with
    the largest shortfall against its share; then every split gets at least one group when
    there are 3 or more (the smallest group of the split with the most groups moves)."""
    total = sum(groups.values())
    have = dict.fromkeys(SPLITS, 0)
    out: dict[str, str] = {}
    for g in sorted(groups, key=lambda g: (-groups[g], g)):
        s = max(SPLITS, key=lambda s: (SHARES[s] * total - have[s], -SPLITS.index(s)))
        out[g], have[s] = s, have[s] + groups[g]
    if len(groups) >= 3:
        for empty in [s for s in reversed(SPLITS) if s not in out.values()]:
            donor = max(SPLITS, key=lambda s: sum(1 for v in out.values() if v == s))
            g = min((g for g in out if out[g] == donor), key=lambda g: (groups[g], g))
            out[g] = empty
    return out


def split(frames: list[dict]) -> dict:
    """frames: [{"file", "source"}]. Returns {"by_frame": {file: split}, "groups": [{group,
    split, frames}], "grouped_by": "site and dive date" | "video", "warning": str | None}."""
    by_site = {f["file"]: site_date(f["source"]) or f["source"] for f in frames}
    grouped_by, warning, key = "site and dive date", None, by_site
    if len(set(by_site.values())) < 3:
        key = {f["file"]: f["source"] for f in frames}
        grouped_by = "video"
        warning = (f"Only {len(set(by_site.values()))} site and dive date group(s): a fair split "
                   "needs at least 3, so frames are split by video instead. Near-identical "
                   "frames from one dive can end up in different splits.")
        if len(set(key.values())) < 3:
            warning += (f" There are only {len(set(key.values()))} video(s), so some splits "
                        "are empty.")
    counts: dict[str, int] = {}
    for g in key.values():
        counts[g] = counts.get(g, 0) + 1
    where = assign(counts)
    return {"by_frame": {f: where[g] for f, g in key.items()},
            "groups": [{"group": g, "split": where[g], "frames": counts[g]}
                       for g in sorted(counts)],
            "grouped_by": grouped_by, "warning": warning}
