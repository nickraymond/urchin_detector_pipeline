"""A project's classes.yaml: the SAM classes, their fixed ids and the SAM settings
(DESIGN, Planned for S3; D6, D18, D22, D25).

The file is read with yaml.safe_load and written with yaml.safe_dump only (D25). Every save
or run is checked first; an error names the line and nothing is saved. A project gets a copy
of app/sam/default_classes.yaml the first time something is saved or run (a POST): a page
load only reads, showing the starting file until then (D14a).

A run unit is a `sam_group` (its classes share one SAM run, and its masks carry no class id
until tagged, D18) or a class with no group. Changing a group's prompts re-runs all of it.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import yaml

DEFAULT_FILE = Path(__file__).with_name("default_classes.yaml")
FILE_NAME = "classes.yaml"
# sam_raw/class_ids.json: {id: name} of every class that has SAM results in this project,
# kept by the SAM runner (bite 2). Such an id keeps its name (D6).
USED_FILE = "class_ids.json"

KINDS = ("object", "area")
MODELS = ("sam3", "sam3.1")
MAX_ID = 254  # the class map is one byte per pixel and 255 marks an untagged urchin (QE PR 24)
CLASS_KEYS = ("id", "name", "kind", "prompts", "sam_group", "seg_as")
# name: (check, help) for the settings block
SETTINGS = {
    "model": (lambda v: v in MODELS, "sam3 or sam3.1"),
    "confidence": (lambda v: _num(v) and 0 < v <= 1, "a number above 0, at most 1"),
    "tile_px": (lambda v: _int(v) and 256 <= v <= 4096, "a whole number from 256 to 4096"),
    "tile_overlap": (lambda v: _num(v) and 0 <= v < 0.9, "a number from 0 to below 0.9"),
    "tile_above_px": (lambda v: _int(v) and v >= 0, "a whole number, 0 or more"),
    "min_mask_area_px": (lambda v: v is None or (_int(v) and v >= 0),
                         "null or a whole number, 0 or more"),
}
# Colours per class id for tables and mask overlays; fixed, so a class keeps its colour.
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#800000",
           "#aaffc3", "#808000", "#ffd8b1", "#000075"]
UNTAGGED_COLOUR = "#808080"  # an urchin mask before P / R (D22 rule 7)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def colour(class_id: int) -> str:
    return PALETTE[(class_id - 1) % len(PALETTE)]


class ClassesError(ValueError):
    """The file has problems; .errors is a list of (line or None, message)."""

    def __init__(self, errors: list[tuple[int | None, str]]):
        self.errors = errors
        super().__init__("; ".join(f"line {n}: {m}" if n else m for n, m in errors))


def file_path(folder: Path) -> Path:
    return Path(folder) / FILE_NAME


def used_ids(folder: Path) -> dict[int, str]:
    """{id: name} of the classes that have SAM results in this project (empty before any run)."""
    p = Path(folder) / "sam_raw" / USED_FILE
    if not p.is_file():
        return {}
    try:
        return {int(k): str(v) for k, v in json.loads(p.read_text()).items()}
    except (ValueError, AttributeError) as e:
        raise ClassesError([(None, f"{p} cannot be read ({e}). It lists the class ids that "
                                   "have SAM results; restore it from a backup.")]) from e


def _lines(text: str) -> dict:
    """1-based line numbers: {"top": {key: n}, "settings": {key: n}, "classes": [{key: n,
    "_": n}]}. Best effort: missing parts are simply absent."""
    out: dict = {"top": {}, "settings": {}, "classes": []}
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return out
    if not isinstance(root, yaml.MappingNode):
        return out
    for k, v in root.value:
        out["top"][k.value] = k.start_mark.line + 1
        if k.value == "settings" and isinstance(v, yaml.MappingNode):
            out["settings"] = {sk.value: sk.start_mark.line + 1 for sk, _ in v.value}
        if k.value == "classes" and isinstance(v, yaml.SequenceNode):
            for item in v.value:
                d = {"_": item.start_mark.line + 1}
                if isinstance(item, yaml.MappingNode):
                    d.update({ck.value: ck.start_mark.line + 1 for ck, _ in item.value})
                out["classes"].append(d)
    return out


def parse(text: str, used: dict[int, str] | None = None) -> dict:
    """Check the text of a classes.yaml; return {"settings": {...}, "classes": [...]}.
    Raises ClassesError with every problem found, each with its line."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        raise ClassesError([(mark.line + 1 if mark else None,
                             f"not valid YAML: {getattr(e, 'problem', None) or e}")]) from e
    lines = _lines(text)
    errs: list[tuple[int | None, str]] = []
    if not isinstance(data, dict):
        raise ClassesError([(1, "the file must have two parts, settings: and classes:")])
    for k in data:
        if k not in ("settings", "classes"):
            errs.append((lines["top"].get(k), f"unknown part {k!r}; only settings: and classes:"))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        errs.append((lines["top"].get("settings", 1), "settings: is missing or not a list of "
                     "name: value lines"))
        settings = {}
    for k, v in settings.items():
        if k not in SETTINGS:
            errs.append((lines["settings"].get(k), f"unknown setting {k!r}; known: "
                         + ", ".join(SETTINGS)))
        elif not SETTINGS[k][0](v):
            errs.append((lines["settings"].get(k), f"{k} must be {SETTINGS[k][1]} (got {v!r})"))
    for k in SETTINGS:
        if data.get("settings") is not None and k not in settings:
            errs.append((lines["top"].get("settings"), f"setting {k} is missing"))
    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        errs.append((lines["top"].get("classes", 1), "classes: is missing or empty"))
        raise ClassesError(errs)
    errs += _check_classes(classes, lines["classes"], used or {})
    if errs:
        raise ClassesError(errs)
    return {"settings": settings, "classes": classes}


def _check_classes(classes: list, lines: list[dict], used: dict[int, str]) -> list:
    errs = []
    ids, names, groups, areas = {}, {}, {}, set()
    free = max([*used, *(c.get("id") for c in classes if isinstance(c, dict)
                         and _int(c.get("id"))), 0]) + 1

    def line(i, key="_"):
        d = lines[i] if i < len(lines) else {}
        return d.get(key, d.get("_"))

    for i, c in enumerate(classes):
        if not isinstance(c, dict):
            errs.append((line(i), "each class must be a list of key: value pairs "
                         "(id, name, kind, prompts)"))
            continue
        for k in c:
            if k not in CLASS_KEYS:
                errs.append((line(i, k), f"unknown key {k!r}; known: " + ", ".join(CLASS_KEYS)))
        cid, name = c.get("id"), c.get("name")
        if not _int(cid) or not 1 <= cid <= MAX_ID:
            errs.append((line(i, "id"), f"id must be a whole number from 1 to {MAX_ID} (0 means "
                         f"unknown, 255 an untagged urchin in the class map); got {cid!r}"))
        elif cid in ids:
            errs.append((line(i, "id"), f"id {cid} is used twice (also line {line(ids[cid])})"))
        else:
            ids[cid] = i
        if not isinstance(name, str) or not name.strip():
            errs.append((line(i, "name"), "every class needs a name"))
            name = None
        elif name.casefold() in names:
            errs.append((line(i, "name"), f"name {name!r} is used twice (also line "
                         f"{line(names[name.casefold()])})"))
        else:
            names[name.casefold()] = i
        if c.get("kind") not in KINDS:
            errs.append((line(i, "kind"), f"kind must be object or area; got {c.get('kind')!r}"))
        elif c["kind"] == "area" and name:
            areas.add(name)
        p = c.get("prompts")
        if not (isinstance(p, list) and p and all(isinstance(s, str) and s.strip() for s in p)):
            errs.append((line(i, "prompts"), "prompts must be a list of at least one text, "
                         "like [sea star]"))
        # ids with SAM results keep their name, and such a name keeps its id (D6)
        if _int(cid) and cid in used and name is not None and used[cid] != name:
            errs.append((line(i, "id"), f"id {cid} has SAM results as {used[cid]!r}; ids are "
                         f"never reused (D6). Give {name!r} a new id, e.g. "
                         f"{free}"))
        old = next((k for k, v in used.items() if name is not None and v == name), None)
        if old is not None and _int(cid) and cid != old:
            errs.append((line(i, "id"), f"{name!r} has SAM results under id {old}; ids are "
                         "never renumbered (D6)"))
        g = c.get("sam_group")
        if g is not None:
            if not isinstance(g, str) or not g.strip():
                errs.append((line(i, "sam_group"), "sam_group must be a name, like urchin"))
            else:
                groups.setdefault(g, []).append(i)
    for g, members in groups.items():
        first = classes[members[0]]
        for i in members:
            if classes[i].get("kind") != "object":
                errs.append((line(i, "kind"), f"classes of sam_group {g!r} must be objects"))
            if classes[i].get("prompts") != first.get("prompts"):
                errs.append((line(i, "prompts"), f"classes of sam_group {g!r} share one SAM "
                             f"run, so they need the same prompts as line {line(members[0])}"))
        if g.casefold() in names and names[g.casefold()] not in members:
            errs.append((line(members[0], "sam_group"), f"sam_group {g!r} has the name of "
                         "another class; pick a different group name"))
    for i, c in enumerate(classes):
        if isinstance(c, dict) and c.get("seg_as") is not None:
            if c.get("kind") != "object":
                errs.append((line(i, "seg_as"), "only an object class can have seg_as"))
            if not isinstance(c["seg_as"], str):
                errs.append((line(i, "seg_as"), f"seg_as must be the name of an area class, "
                             f"like kelp (got {c['seg_as']!r})"))
            elif c["seg_as"] not in areas:
                errs.append((line(i, "seg_as"), f"seg_as must name an area class in this file "
                             f"(got {c['seg_as']!r})"))
    return errs


def run_units(classes: list[dict]) -> list[dict]:
    """One entry per SAM run, in file order: {"key": group or class name, "group": bool,
    "classes": {id: name}, "kind", "prompts"}."""
    units: dict[str, dict] = {}
    for c in classes:
        key = c.get("sam_group") or c["name"]
        u = units.setdefault(key, {"key": key, "group": bool(c.get("sam_group")),
                                   "classes": {}, "kind": c["kind"], "prompts": c["prompts"]})
        u["classes"][c["id"]] = c["name"]
    return list(units.values())


def read(folder: Path) -> dict:
    """For the page: {"path", "exists", "text", "data" (None if it has errors), "errors",
    "used"}. Never writes; a project with no classes.yaml yet shows the starting file."""
    p = file_path(folder)
    exists = p.is_file()
    text = (p if exists else DEFAULT_FILE).read_text()
    out = {"path": p, "exists": exists, "text": text, "data": None, "errors": [], "used": {}}
    try:
        out["used"] = used_ids(folder)
        out["data"] = parse(text, out["used"])
    except ClassesError as e:
        out["errors"] = e.errors
    return out


def _write(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def save(folder: Path, text: str) -> dict:
    """Check `text` and write it as classes.yaml (safe_dump of the checked data, D25).
    Raises ClassesError and writes nothing if it has problems."""
    data = parse(text, used_ids(folder))
    _write(file_path(folder), dump(data))
    return data


def dump(data: dict) -> str:
    """The same layout as the starting file: settings one per line, one class per line.
    Every piece is written by yaml.safe_dump (D25)."""
    return "".join(line for line, _ in _dump_lines(data))


def _dump_lines(data: dict) -> list[tuple[str, tuple]]:
    """dump()'s lines, each with where it came from: ("settings", key), ("class", row) or ()."""
    def one(value, flow):
        return yaml.safe_dump(value, sort_keys=False, allow_unicode=True,
                              default_flow_style=flow, width=10_000).strip()
    out: list[tuple[str, tuple]] = [("settings:\n", ())]
    st = data["settings"]   # in the starting file's order (the page's JSON sorts keys; QE PR 28)
    for k in [*(k for k in SETTINGS if k in st), *(k for k in st if k not in SETTINGS)]:
        out += [("  " + line + "\n", ("settings", k)) for line in one({k: st[k]}, False).splitlines()]
    out.append(("classes:\n", ()))
    for i, c in enumerate(data["classes"]):
        out += [(("  - " if n == 0 else "    ") + line + "\n", ("class", i))
                for n, line in enumerate(one(c, True).splitlines())]
    return out


# --- The class form (S3 bite 6, SPEC Q17): the page sends {settings, classes} as JSON ---

def form_check(folder: Path, data) -> tuple[str | None, list[dict]]:
    """The form's data as classes.yaml text, checked with the same checks as a text save.
    Returns (text, []) when it is fine, else (None, [{"row", "setting", "msg"}]): row is the
    class's place in the form, setting a settings name, both None for the whole file."""
    if not (isinstance(data, dict) and isinstance(data.get("settings"), dict)
            and isinstance(data.get("classes"), list)):
        return None, [{"row": None, "setting": None,
                       "msg": "the form sent no settings and classes; reload the page"}]
    lines = _dump_lines({"settings": data["settings"], "classes": data["classes"]})
    text = "".join(line for line, _ in lines)
    try:
        parse(text, used_ids(folder))
    except ClassesError as e:
        out = []
        for n, msg in e.errors:
            where = lines[n - 1][1] if n and 0 < n <= len(lines) else ()
            out.append({"row": where[1] if where[:1] == ("class",) else None,
                        "setting": where[1] if where[:1] == ("settings",) else None,
                        "msg": msg})
        return None, out
    return text, []


def form_save(folder: Path, data) -> list[dict]:
    """Save the form's classes as classes.yaml if they pass every check; else write nothing
    and return the problems (form_check)."""
    text, errors = form_check(folder, data)
    if errors:
        return errors
    save(folder, text)
    return []


def ensure(folder: Path) -> dict:
    """Before a run (a POST): copy the starting file into the project if it has none, then
    load and check it. Raises ClassesError."""
    p = file_path(folder)
    if not p.is_file():
        tmp = p.with_name(p.name + ".tmp")
        shutil.copyfile(DEFAULT_FILE, tmp)
        os.replace(tmp, p)
    return parse(p.read_text(), used_ids(folder))
