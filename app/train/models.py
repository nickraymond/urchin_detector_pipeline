"""A project's models folder (spec §Outputs; DESIGN D53, D54 from S6 bite 1):

    <project>/models/m<number>-<arch key>/
        recipe.json          {"arch": {...}, "train": {...}}; the train job adds "recorded"
        checkpoints/         last.* (where Resume continues) and what the trainer keeps
        metrics.jsonl        one line per finished epoch

Model ids are numbered per project and never reused; the app never removes a model folder.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MODEL_DIR = re.compile(r"^m(\d+)-([a-z0-9][a-z0-9.-]*)$")


class ModelError(ValueError):
    """A problem with a model folder or id; the message says what."""


def root(folder: Path) -> Path:
    return Path(folder) / "models"


def write_json(p: Path, data) -> None:
    """Under a temporary name, then renamed: a killed worker never leaves half a file."""
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, p)


def list_ids(folder: Path) -> list[str]:
    top = root(folder)
    ids = [p.name for p in top.iterdir() if MODEL_DIR.match(p.name)] if top.is_dir() else []
    return sorted(ids, key=lambda i: int(MODEL_DIR.match(i)[1]))


def create(folder: Path, arch_key: str, recipe: dict) -> Path:
    """Make the next model folder with its recipe. arch_key: e.g. "yolox-nano-conv-256-c5"."""
    if not MODEL_DIR.match(f"m1-{arch_key}"):
        raise ModelError(f"Bad architecture key {arch_key!r}: lower-case letters, digits, "
                         "'.' and '-' only.")
    numbers = [int(MODEL_DIR.match(i)[1]) for i in list_ids(folder)]
    mdir = root(folder) / f"m{max(numbers, default=0) + 1}-{arch_key}"
    (mdir / "checkpoints").mkdir(parents=True)
    write_json(mdir / "recipe.json", recipe)
    return mdir


def model_dir(folder: Path, model_id: str) -> Path:
    """The folder of an existing model; refuses anything that is not a model id."""
    if not MODEL_DIR.match(model_id or "") or not (root(folder) / model_id).is_dir():
        raise ModelError(f"No model {model_id!r} in {root(folder)}.")
    return root(folder) / model_id


def read_recipe(mdir: Path) -> dict:
    try:
        return json.loads((Path(mdir) / "recipe.json").read_text())
    except (OSError, ValueError) as e:
        raise ModelError(f"{mdir}/recipe.json cannot be read ({e}).") from None


def read_metrics(mdir: Path) -> list[dict]:
    """The finished epochs' metrics lines, in order. A torn last line is ignored."""
    p = Path(mdir) / "metrics.jsonl"
    out = []
    for line in p.read_text().splitlines() if p.is_file() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            break
    return out
