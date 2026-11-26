"""Label review page glue (S5 bite 3a; spec §6, D40, D41, SPEC Q16).

The page lists the labeled kept frames with accepted / to do and opens them in S2's large
viewer. The picture is drawn here from the frame's final masks (sam_raw + review, after the
overlap rules: app/review/state.py), with the selected mask outlined in white. A click on the
picture asks `mask_at` which mask is under that pixel (objects before areas, the smallest
object first, so a fish on a rock is picked, not the rock). Actions go to app/review/state.py.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter

from app.review import state, tagging
from app.sam import classes, overlap, results
from app.stages import extract, label

MODES = ("all", "objects", "areas", "map")
WIDTH = label.LARGE_WIDTH        # the viewer's picture; smaller frames stay as they are


def frame_list(folder: Path, conn) -> list[dict]:
    """Kept frames with SAM results, in name order: {file, accepted}."""
    out = []
    for f in label.frames_to_label(conn):
        if results.raw_path(folder, f).is_file():
            out.append({"file": f, "accepted": state.frame_state(conn, f)["accepted"]})
    return out


def colour_of(class_id: int | None) -> str:
    return classes.UNTAGGED_COLOUR if class_id is None else classes.colour(class_id)


def view(folder: Path, conn, frame: str, class_list: list[dict]) -> dict:
    """What the page needs for one frame: its size, state, and every mask shown on it."""
    fin = state.final(folder, conn, frame, class_list)
    names = {c["id"]: c["name"] for c in class_list}
    masks = [{"id": o["mask_id"], "kind": "object", "class_id": o["class_id"],
              "name": names.get(o["class_id"], "urchin (untagged)"),
              "colour": colour_of(o["class_id"]), "box": o["box"], "score": o["score"],
              "flag": o["flag"], "reviewed": o["reviewed"], "older": o["from_older_settings"],
              "added": o["added"]} for o in fin["objects"]]
    masks += [{"id": a["mask_id"], "kind": "area", "class_id": a["class_id"],
               "name": names.get(a["class_id"], "?"), "colour": colour_of(a["class_id"]),
               "box": a["box"], "reviewed": a["reviewed"]} for a in fin["areas"]]
    types = tagging.urchin_types(class_list)
    groups = {t["group"] for t in types}
    type_ids = {t["id"] for t in types}
    for m in masks:   # bite 4b: which masks P / R / B can tag, and where Tab zooms to
        if m["kind"] == "object":
            o = next(o for o in fin["objects"] if o["mask_id"] == m["id"])
            m["urchin"] = o["unit"] in groups or o["class_id"] in type_ids
            m["focus"] = tagging.focus(o["box"], fin["width"], fin["height"])
    order = [o["mask_id"] for o in sorted(fin["objects"],
                                          key=lambda o: (o["box"][1], o["box"][0], o["mask_id"]))]
    return {"frame": frame, "width": fin["width"], "height": fin["height"],
            "state": state.frame_state(conn, frame), "untagged": fin["untagged"],
            "masks": masks, "types": types, "order": order}


def _rles(fin: dict) -> dict[int, dict]:
    return {m["mask_id"]: m for u in fin["raw"]["units"].values() for m in u["masks"]}


def mask_at(folder: Path, conn, frame: str, class_list: list[dict], x: int, y: int) -> int | None:
    """The mask under frame pixel (x, y): objects first, the smallest first; then areas."""
    fin = state.final(folder, conn, frame, class_list)
    rles = _rles(fin)

    def hit(m):
        x0, y0, x1, y1 = m["box"]
        return (x0 <= x < x1 and y0 <= y < y1
                and bool(results.decode_box(rles[m["mask_id"]]["rle"], m["box"])[y - y0, x - x0]))
    for o in sorted(fin["objects"], key=lambda o: o["area"]):
        if hit(o):
            return o["mask_id"]
    for a in sorted(fin["areas"], key=lambda a: a["area"]):
        if hit(a):
            return a["mask_id"]
    return None


CLOSEUP_PX = 480


def closeup(folder: Path, conn, frame: str, class_list: list[dict], mask_id: int) -> bytes | None:
    """The close-up panel (bite 4b): the frame at full resolution around one mask (its focus
    rectangle), CLOSEUP_PX wide, with only that mask's outline, so its colour is easy to judge."""
    path = extract.find_file(folder, frame)
    if path is None or not results.raw_path(folder, frame).is_file():
        return None
    fin = state.final(folder, conn, frame, class_list)
    m = next((o for o in fin["objects"] + fin["areas"] if o["mask_id"] == mask_id), None)
    if m is None:
        return None
    st = path.stat()
    full = label._scaled_frame(str(path), (st.st_mtime_ns, st.st_size), 100_000)  # never up: full size
    x0, y0, x1, y1 = tagging.focus(m["box"], fin["width"], fin["height"])
    im = full.crop((x0, y0, x1, y1))
    s = CLOSEUP_PX / im.width
    im = im.resize((CLOSEUP_PX, max(1, round(im.height * s))), Image.BILINEAR).convert("RGBA")
    bx0, by0, bx1, by1 = m["box"]
    crop = results.decode_box(_rles(fin)[mask_id]["rle"], m["box"])
    wh = (max(1, round((bx1 - bx0) * s)), max(1, round((by1 - by0) * s)))
    mk = Image.fromarray(crop.astype(np.uint8) * 255).resize(wh)
    edge = mk.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    layer.paste(Image.new("RGBA", wh, (255, 255, 255, 230)),
                (round((bx0 - x0) * s), round((by0 - y0) * s)), edge)
    out = io.BytesIO()
    Image.alpha_composite(im, layer).convert("RGB").save(out, "JPEG", quality=88)
    return out.getvalue()


def picture(folder: Path, conn, frame: str, class_list: list[dict], mode: str = "all",
            selected: int | None = None, width: int = WIDTH) -> bytes | None:
    """The frame scaled to `width` with its final masks: areas faint, objects on top with a
    box; `mode` objects / areas shows one kind, map the final class map. The selected mask
    gets a white outline. JPEG bytes, None if the frame or its results are missing."""
    path = extract.find_file(folder, frame)
    if path is None or not results.raw_path(folder, frame).is_file():
        return None
    st = path.stat()
    im = label._scaled_frame(str(path), (st.st_mtime_ns, st.st_size), width)
    fin = state.final(folder, conn, frame, class_list)
    sx, sy = im.width / fin["width"], im.height / fin["height"]
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rles = _rles(fin)

    def scaled(m):
        x0, y0, x1, y1 = m["box"]
        bx0, by0 = round(x0 * sx), round(y0 * sy)
        wh = (max(1, round(x1 * sx) - bx0), max(1, round(y1 * sy) - by0))
        crop = results.decode_box(rles[m["mask_id"]]["rle"], m["box"])
        return (bx0, by0), wh, Image.fromarray(crop.astype(np.uint8) * 255).resize(wh)

    line = max(1, round(im.width / 1600))
    if mode == "map":
        cmap = state.class_map(fin, class_list)
        small = Image.fromarray(cmap).resize(im.size, Image.NEAREST)
        lut = [(0, 0, 0, 0)] * 256
        for c in class_list:
            lut[c["id"]] = (*ImageColor.getrgb(classes.colour(c["id"])), 140)
        lut[overlap.UNTAGGED] = (*ImageColor.getrgb(classes.UNTAGGED_COLOUR), 140)
        a = np.asarray(small)
        layer = Image.fromarray(np.array(lut, np.uint8)[a], "RGBA")
        draw = ImageDraw.Draw(layer)
    else:
        if mode in ("all", "areas"):
            for m in fin["areas"]:
                pos, wh, mk = scaled(m)
                rgb = ImageColor.getrgb(colour_of(m["class_id"]))
                layer.paste(Image.new("RGBA", wh, (*rgb, 90)), pos, mk)
        if mode in ("all", "objects"):
            for o in fin["objects"]:
                pos, wh, mk = scaled(o)
                rgb = ImageColor.getrgb(colour_of(o["class_id"]))
                layer.paste(Image.new("RGBA", wh, (*rgb, 150)), pos, mk)
                draw.rectangle([pos[0], pos[1], pos[0] + wh[0], pos[1] + wh[1]],
                               outline=(*rgb, 255), width=line)
    sel = next((m for m in fin["objects"] + fin["areas"] if m["mask_id"] == selected), None)
    if sel is not None:
        pos, wh, mk = scaled(sel)
        edge = mk.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(2 * line + 1))
        layer.paste(Image.new("RGBA", wh, (255, 255, 255, 255)), pos, edge)
        draw.rectangle([pos[0] - 2 * line, pos[1] - 2 * line, pos[0] + wh[0] + 2 * line,
                        pos[1] + wh[1] + 2 * line], outline=(255, 255, 255, 255), width=2 * line)
    out = io.BytesIO()
    Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB").save(out, "JPEG", quality=85)
    return out.getvalue()


_TOTALS: dict[tuple, dict] = {}   # (folder, frame) -> (version, {class id: objects}); bite 3b


def _frame_counts(folder: Path, conn, frame: str, class_list: list[dict]) -> dict[int, int]:
    """Objects per class on one frame after review, cached until its results or review change."""
    p = results.raw_path(folder, frame)
    st = p.stat()
    version = (st.st_mtime_ns, st.st_size, repr(sorted(state.mask_rows(conn, frame).items())),
               repr([(c["id"], c["kind"]) for c in class_list]))
    hit = _TOTALS.get((str(folder), frame))
    if hit and hit[0] == version:
        return hit[1]
    counts: dict[int, int] = {}
    for o in state.final(folder, conn, frame, class_list)["objects"]:
        if o["class_id"] is not None:
            counts[o["class_id"]] = counts.get(o["class_id"], 0) + 1
    _TOTALS[(str(folder), frame)] = (version, counts)
    return counts


def totals(folder: Path, conn, class_list: list[dict]) -> list[dict]:
    """Running totals per object class: {id, name, colour, objects, frames, split_no}. objects
    and frames count the accepted frames that go to the detector (split check not "no", spec
    §6), which the page compares with the 150-200 target; split_no = objects on accepted frames
    marked "no" (they count for cover only; QE PR 51)."""
    rows = conn.execute("SELECT frame, split_ok FROM frame_review WHERE accepted = 1").fetchall()
    per = [(_frame_counts(folder, conn, f, class_list), ok != 0) for f, ok in rows
           if results.raw_path(folder, f).is_file()]
    return [{"id": c["id"], "name": c["name"], "colour": classes.colour(c["id"]),
             "objects": sum(p.get(c["id"], 0) for p, det in per if det),
             "frames": sum(1 for p, det in per if det and p.get(c["id"])),
             "split_no": sum(p.get(c["id"], 0) for p, det in per if not det)}
            for c in class_list if c["kind"] == "object"]


def act(folder: Path, conn, frame: str, class_list: list[dict], body: dict) -> str:
    """One review action from the page; returns a short note. Raises state.ReviewError."""
    a = body.get("action")
    raw_ids = body.get("mask_ids") or []
    if not isinstance(raw_ids, list) or not all(isinstance(i, int) for i in raw_ids):
        raise state.ReviewError("Bad request: mask_ids must be a list of numbers.")
    ids = list(raw_ids)
    if a == "class":
        cid = body.get("class_id")
        if not isinstance(cid, int):
            raise state.ReviewError("Pick a class from the list.")
        state.set_class(conn, folder, frame, ids, cid, class_list)
        return "class changed"
    if a == "delete":
        state.delete(conn, folder, frame, ids)
        return "deleted"
    if a == "accept":
        state.accept(conn, folder, frame, class_list)
        return "frame accepted"
    if a == "reopen":
        state.reopen(conn, frame)
        return "frame back to do"
    if a == "split":
        ok = body.get("ok")
        if ok not in (True, False, None):
            raise state.ReviewError("Split check is yes, no or not marked.")
        state.set_split(conn, frame, ok)
        return "split check " + ("not marked" if ok is None else "yes" if ok else "no")
    if a == "checked":
        cid = body.get("class_id")
        if not isinstance(cid, int):
            raise state.ReviewError("Pick a class.")
        state.set_checked(conn, frame, cid, bool(body.get("checked")), class_list)
        return "fully checked: " + ("on" if body.get("checked") else "off")
    if a == "tag":
        cid = body.get("class_id")
        if not isinstance(cid, int):
            raise state.ReviewError("Pick an urchin type.")
        tagging.tag(conn, folder, frame, ids, cid, class_list)
        return "tagged " + next(t["name"] for t in tagging.urchin_types(class_list) if t["id"] == cid)
    if a == "tag_all":
        cid = body.get("class_id")
        if not isinstance(cid, int):
            raise state.ReviewError("Pick an urchin type.")
        n = tagging.tag_all_remaining(conn, folder, frame, cid, class_list)
        name = next(t["name"] for t in tagging.urchin_types(class_list) if t["id"] == cid)
        return f"tagged {n} remaining urchin{'s' if n > 1 else ''} {name}"
    if a == "undo":
        done = state.undo(conn, frame)
        return f"undone: {done.split(' on ')[0]}" if done else "nothing to undo on this frame"
    raise state.ReviewError(f"Unknown action {a!r}.")
