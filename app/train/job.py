"""The app side of the "train" job kind (S8 bite 4a): which trainers exist, the job file,
and progress counted from the model's own files."""

from __future__ import annotations

from pathlib import Path

from app.train import models

# name -> "module:Class". Only names listed here can run: the job file never carries a module
# path that came from a web form. Tests add their fake trainer here.
TRAINERS: dict[str, str] = {}


def prepare(params: dict) -> dict:
    name = params.get("trainer")
    if name not in TRAINERS:
        raise ValueError(f"No trainer {name!r}. Known: {', '.join(sorted(TRAINERS)) or 'none'}.")
    return {**params, "trainer": TRAINERS[name]}


def progress(params: dict) -> tuple[int, int]:
    """(epochs with a metrics line, epochs asked for), from the model folder."""
    mdir = Path(params["model_dir"])
    return len(models.read_metrics(mdir)), int(models.read_recipe(mdir)["train"]["epochs"])


def train_kind():
    from app.jobctl import Kind
    return Kind("train", module="app.train.worker", prepare=prepare, progress=progress,
                item="epoch")
