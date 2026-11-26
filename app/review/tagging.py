"""Urchin types and fast tagging (S5 bite 4; D18, D39).

SAM runs one prompt for every urchin type; its masks have no class until tagged. Each class of a
`sam_group` is a type: purple, red and black urchin in the starting file. A type's key is the
first letter of its name (P, R, B), so a renamed type gets its new letter; a letter that clashes
with another type or a review key gets none (the class picker still works).

Most urchins are purple (Nick, 2026-11-25), so review finds the outliers: `next_untagged` gives
the next untagged urchin in reading order for `Tab`, `focus` the part of the frame the viewer
zooms to (the urchin with some scene around it), and `tag_all_remaining` tags every untagged
urchin on the frame at once (one undo step).
"""

from __future__ import annotations

from pathlib import Path

from app.review import state

RESERVED = set("XZF")            # label review keys: X delete, Z undo, F full screen
FOCUS_SCALE = 3.0                # the zoomed view is 3x the urchin's size ...
FOCUS_MIN_PX = 400               # ... and at least this wide and high


def urchin_types(class_list: list[dict]) -> list[dict]:
    """[{"id", "name", "group", "key" (a letter or None)}] for every class in a sam_group."""
    types = [c for c in class_list if c.get("sam_group")]
    letters = [c["name"].strip()[:1].upper() for c in types]
    return [{"id": c["id"], "name": c["name"], "group": c["sam_group"],
             "key": k if k.isalpha() and k not in RESERVED and letters.count(k) == 1 else None}
            for c, k in zip(types, letters)]


def _order(objs: list[dict]) -> list[dict]:
    """Reading order: top to bottom by the box's top edge, then left to right."""
    return sorted(objs, key=lambda o: (o["box"][1], o["box"][0], o["mask_id"]))


def next_untagged(fin: dict, after: int | None = None) -> int | None:
    """The untagged urchin after mask `after` in reading order, wrapping round; None if the
    frame has none. `fin` is state.final()."""
    todo = _order([o for o in fin["objects"] if o["class_id"] is None])
    if not todo:
        return None
    ids = [o["mask_id"] for o in todo]
    if after is None:
        return ids[0]
    allo = _order(fin["objects"])
    pos = next((i for i, o in enumerate(allo) if o["mask_id"] == after), None)
    if pos is None:
        return ids[0]
    later = [o["mask_id"] for o in allo[pos + 1:] if o["class_id"] is None]
    return later[0] if later else ids[0]


def focus(box: list[int], width: int, height: int) -> list[int]:
    """[x0, y0, x1, y1] to zoom on: the box's centre, FOCUS_SCALE times its size (at least
    FOCUS_MIN_PX), with the frame's aspect, kept inside the frame."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * FOCUS_SCALE, (y1 - y0) * FOCUS_SCALE * width / height, FOCUS_MIN_PX)
    w = min(w, width)
    h = min(w * height / width, height)
    left = min(max(cx - w / 2, 0), width - w)
    top = min(max(cy - h / 2, 0), height - h)
    return [round(left), round(top), round(left + w), round(top + h)]


def _type(class_id: int, class_list: list[dict]) -> dict:
    t = next((t for t in urchin_types(class_list) if t["id"] == class_id), None)
    if t is None:
        raise state.ReviewError(f"Class {class_id} is not an urchin type (a class of a group).")
    return t


def tag(conn, folder: Path, frame: str, mask_ids: list[int], class_id: int,
        class_list: list[dict]) -> None:
    """`P` / `R` / `B`: tag the selected urchins with a type. Only urchins (masks of that
    type's group, or already tagged with one of its types) can be tagged."""
    t = _type(class_id, class_list)
    fin = state.final(folder, conn, frame, class_list)
    group_ids = {x["id"] for x in urchin_types(class_list) if x["group"] == t["group"]}
    by_id = {o["mask_id"]: o for o in fin["objects"]}
    for mid in mask_ids:
        o = by_id.get(mid)
        if o is None or not (o["unit"] == t["group"] or o["class_id"] in group_ids):
            raise state.ReviewError(f"Mask {mid} is not an urchin; use the class picker to "
                                    "change its class.")
    state.set_class(conn, folder, frame, mask_ids, class_id, class_list)


def tag_all_remaining(conn, folder: Path, frame: str, class_id: int,
                      class_list: list[dict]) -> int:
    """Tag every untagged urchin on the frame with one type (one undo step). Returns how many."""
    _type(class_id, class_list)
    todo = state.final(folder, conn, frame, class_list)["untagged"]
    if not todo:
        raise state.ReviewError("No untagged urchin left on this frame.")
    state.set_class(conn, folder, frame, todo, class_id, class_list)
    return len(todo)
