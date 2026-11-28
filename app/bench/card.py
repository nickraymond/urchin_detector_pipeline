"""Test cards (D84): a bench run's question and rules, written before the run and saved with it.

    <data_dir>/bench/cards/<n>.json          the card; never changed once saved (a change is a new card)
    <data_dir>/bench/cards/<n>.review.json   who reviewed it, against the card file's sha256

The lead writes a card, QE reviews it, and only a reviewed card can be run (bite 2b refuses the rest).
Model files must lie under the data folder; their sha256 is taken when the card is saved, so a file that
changes afterwards is caught before the run.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.bench import units

MIN_WINDOWS, DEFAULT_WINDOWS, DEFAULT_N = 3, 3, 300           # D83: never a cell from one window
DEFAULT_METRIC = "ms per inference (median and spread over the windows); free memory before load, after load, after"
DEFAULT_STOP_RULES = [                                          # D84, D87
    "a board that does not answer in 60 s stops the run",
    "a load whose read-back differs stops the run; that board is put back from the backup before anything else",
    "on the AE3: one operation per call, no retries",
    "RESTORE.md matches the rig log",
]
TEXT_MAX = 2000


class CardError(ValueError):
    """A card that cannot be saved or reviewed; the message lists every problem."""


def folder(data_dir) -> Path:
    return Path(data_dir) / "bench" / "cards"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(data_dir, bench: units.Bench, form: dict) -> dict:
    """A card from the page's form, checked; raises CardError with every problem at once."""
    problems = []
    text = {k: " ".join(str(form.get(k, "")).split()) for k in ("question", "pass_bar", "written_by", "metric")}
    for k, label in (("question", "the question"), ("pass_bar", "the pass bar"), ("written_by", "who wrote it")):
        if not text[k]:
            problems.append(f"{label} is empty")
    if any(len(v) > TEXT_MAX for v in text.values()):
        problems.append(f"a text field is longer than {TEXT_MAX} characters")
    unit_name, board = str(form.get("unit", "")).strip(), str(form.get("board", "")).strip()
    try:
        unit = bench.unit(unit_name)
        if board not in unit.kinds:
            problems.append(f"unit {unit_name} has no {board or '(none)'} board (it has {', '.join(unit.kinds)})")
    except units.Refused as e:
        problems.append(str(e))
    windows, n = _int(form.get("windows"), DEFAULT_WINDOWS), _int(form.get("n"), DEFAULT_N)
    if windows is None or windows < MIN_WINDOWS:
        problems.append(f"windows must be a whole number of at least {MIN_WINDOWS} (D83: no cell from one window)")
    if n is None or n < 1:
        problems.append("N (inferences per window) must be a whole number of at least 1")
    cost = _int(form.get("cost_minutes"), None)
    if cost is None or cost < 1:
        problems.append("the cost (minutes on the rig) must be a whole number of at least 1")
    models = _models(data_dir, form.get("models", ""), problems)
    rules = [" ".join(r.split()) for r in str(form.get("stop_rules", "")).splitlines() if r.strip()]
    if not rules:
        problems.append("a card needs at least one stop rule")
    if problems:
        raise CardError("; ".join(problems) + ".")
    return {"question": text["question"], "unit": unit_name, "board": board, "models": models,
            "frames": "one seeded synthetic image at each model's input size, sent as a file (D83)",
            "windows": windows, "n": n, "metric": text["metric"] or DEFAULT_METRIC, "pass_bar": text["pass_bar"],
            "stop_rules": rules, "cost_minutes": cost, "written_by": text["written_by"], "written_at": _now()}


def _int(value, default):
    if value in (None, ""):
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _models(data_dir, lines: str, problems: list) -> list[dict]:
    root, out, names = Path(data_dir).resolve(), [], set()
    for line in str(lines).splitlines():
        if not line.strip():
            continue
        p = Path(line.strip()).expanduser()
        p = (p if p.is_absolute() else root / p).resolve()
        if not p.is_relative_to(root):
            problems.append(f"{line.strip()} is not under the data folder {root}")
        elif not p.is_file():
            problems.append(f"model file not found: {p}")
        elif _name(root, p) in names:
            problems.append(f"two model files are named {_name(root, p)}")
        else:
            names.add(_name(root, p))
            out.append({"name": _name(root, p), "path": str(p), "bytes": p.stat().st_size, "sha256": _sha(p)})
    if not out and not any("model file" in x or "data folder" in x for x in problems):
        problems.append("list at least one model file, one path per line")
    return out


def _name(root: Path, p: Path) -> str:
    """A model's name on the card: the candidate's folder for S6's files (`fit/<arch key>/int8/model_int8.tflite`
    is "<arch key>", since every candidate's file has the same name), else its path under the data folder."""
    rel = p.relative_to(root)
    return rel.parts[1] if len(rel.parts) > 2 and rel.parts[0] == "fit" else str(rel.with_suffix(""))


def save(data_dir, card: dict) -> int:
    """Write the card under the next number; never overwrites (O_EXCL)."""
    d = folder(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    n = max([int(p.stem) for p in d.glob("*.json") if p.stem.isdigit()] + [0]) + 1
    while True:
        try:
            fd = os.open(d / f"{n}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            break
        except FileExistsError:
            n += 1
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({**card, "number": n}, f, indent=1, sort_keys=True)
    return n


def load(data_dir, number: int) -> dict:
    """The card with its review state: `sha256` of the card file, `review` (or None), `reviewed` (bool)."""
    p = folder(data_dir) / f"{int(number)}.json"
    if not p.is_file():
        raise CardError(f"No test card {number}.")
    card = json.loads(p.read_text(encoding="utf-8"))
    card["sha256"] = _sha(p)
    r = folder(data_dir) / f"{int(number)}.review.json"
    card["review"] = json.loads(r.read_text(encoding="utf-8")) if r.is_file() else None
    card["reviewed"] = bool(card["review"] and card["review"].get("card_sha256") == card["sha256"])
    return card


def list_cards(data_dir) -> list[dict]:
    d = folder(data_dir)
    nums = sorted((int(p.stem) for p in d.glob("*.json") if p.stem.isdigit()), reverse=True) if d.is_dir() else []
    return [load(data_dir, n) for n in nums]


def review(data_dir, number: int, reviewed_by: str) -> dict:
    """Record a review of the card as it is on disk now. The writer cannot review their own card."""
    card, who = load(data_dir, number), " ".join(str(reviewed_by).split())
    if not who:
        raise CardError("Say who reviewed it.")
    if who.lower() == card["written_by"].lower():
        raise CardError("A card is reviewed by someone other than its writer (D84: the lead writes, QE reviews).")
    if card["review"]:
        raise CardError(f"Card {number} is already reviewed by {card['review']['reviewed_by']}.")
    body = {"reviewed_by": who, "reviewed_at": _now(), "card_sha256": card["sha256"]}
    fd = os.open(folder(data_dir) / f"{int(number)}.review.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=1, sort_keys=True)
    return load(data_dir, number)
