"""The app side of the "train" job kind (S8 bite 4a): which trainers exist, the job file,
and progress counted from the model's own files."""

from __future__ import annotations

import json
from pathlib import Path

from app.datasets import loaders, outside
from app.train import models

# name -> "module:Class". Only names listed here can run: the job file never carries a module
# path that came from a web form. Tests add their fake trainer here.
TRAINERS: dict[str, str] = {}


def prepare(params: dict) -> dict:
    name = params.get("trainer")
    if name not in TRAINERS:
        raise ValueError(f"No trainer {name!r}. Known: {', '.join(sorted(TRAINERS)) or 'none'}.")
    return {**params, "trainer": TRAINERS[name]}


ESTIMATE_STEPS = 5               # training steps an Estimate times, after 1 warm-up step
ESTIMATE_DIR = ".estimate"       # models/.estimate/: the recipe being estimated + estimate.json


def estimate_dir(folder: Path) -> Path:
    return models.root(folder) / ESTIMATE_DIR


def read_estimate(folder: Path) -> dict | None:
    """The last Estimate of this project (app/train/worker.py `estimate`), or None."""
    try:
        return json.loads((estimate_dir(folder) / "estimate.json").read_text())
    except (OSError, ValueError):
        return None


def progress(params: dict) -> tuple[int, int]:
    """(epochs with a metrics line, epochs asked for), from the model folder."""
    if params.get("estimate"):
        return 0, 1
    mdir = Path(params["model_dir"])
    return len(models.read_metrics(mdir)), int(models.read_recipe(mdir)["train"]["epochs"])


def train_kind():
    from app.jobctl import Kind
    return Kind("train", module="app.train.worker", prepare=prepare, progress=progress,
                item="epoch")


# ---- model types the Train page offers (bite 5); bite 4b adds YOLOX here ----
# key -> {"name", "kind", "trainer" (a key of TRAINERS), "family", "size", "variant" (str),
#         "inputs": [px, ...], "presets": [augmentation presets], "defaults": {"epochs", "input",
#         "augment", "seed"}, "why": {field: one short sentence}, "licence"}
MODEL_TYPES: dict[str, dict] = {}

# YOLOX (bite 4b): S6's definition (app/modeldefs/yolox.py), trained by app/train/yolox.py.
# Defaults are the earlier work's known-good recipe (ADIN_SPI_OpenMV ml/yolox_urchin/STAGE1.md).
TRAINERS["yolox"] = "app.train.yolox:Trainer"
_YOLOX_WHY = {
    "input": "256 px is the size that compiled for both camera chips and was timed on them; "
             "training tiles are cut so urchins are 24-64 px, as the cameras see them.",
    "epochs": "The earlier Nano run kept improving to 120 epochs (mAP50 0.44 at epoch 20, 0.65 "
              "at 120, on public urchin data).",
    "val_every": "Validation scores whole frames in tiles (a 4K frame can be hundreds of tiles), so it can "
                 "take as long as an epoch; the last epoch is always validated, and best.pt is picked "
                 "from validated epochs only (Lead call, D90). 1 = every epoch.",
    "augment": "Blur and haze stand in for turbid water; haze is new and not yet measured "
               "(the earlier work found it hurt Nano), so compare a run without it."}
for _size, _note in (("nano", "smallest; 10.6 ms on the N6 and 26.4 ms on the AE3 in the earlier "
                              "work."),
                     ("tiny", "more accurate in the earlier work (mAP50 0.73 against 0.65), about "
                              "3x slower on the N6; on the AE3 it runs from the model area only.")):
    MODEL_TYPES[f"yolox-{_size}-conv"] = {
        "name": f"YOLOX-{_size.capitalize()} (plain first layer)", "kind": "detector",
        "trainer": "yolox", "modeldef": "app.modeldefs.yolox", "family": "yolox", "size": _size,
        "variant": "conv", "inputs": [256, 320],
        "presets": ["standard + blur + haze", "standard", "none"],
        "defaults": {"epochs": 120, "input": 256, "augment": "standard + blur + haze", "seed": 0,
                     "val_every": 5},
        "starts": ("scratch", "pretrained"), "outside": True,
        "why": {"type": _note, **_YOLOX_WHY}, "licence": "Apache-2.0"}


usable = loaders.usable      # the one frame rule of a detector run (app/datasets/loaders.py)


def arch_key(mtype: dict, input_px: int, n_classes: int) -> str:
    """D53: <family>-<size>[-<variant>]-<input px>-c<classes>, e.g. yolox-nano-conv-256-c5."""
    parts = [mtype["family"], mtype["size"]] + ([mtype["variant"]] if mtype.get("variant") else [])
    return "-".join(parts + [str(input_px), f"c{n_classes}"])


def build_recipe(mtype_key: str, form: dict, manifest: dict, class_ids: list[int],
                 start: dict) -> tuple[str, dict]:
    """(arch key, recipe.json) for a new model (D54) from the Train page's form. Raises
    ValueError with a message for the page when a value is not allowed."""
    mtype = MODEL_TYPES.get(mtype_key)
    if mtype is None:
        raise ValueError("Pick a model type.")
    if start.get("kind") not in mtype.get("starts", ("scratch",)):
        raise ValueError(f"{mtype['name']} can only start from scratch for now: its trainer does "
                         "not load starting weights yet.")
    data = form.get("data") or "project"
    if data not in ("project", "outside"):
        raise ValueError("Train on: this project's frames, or the version's outside data.")
    if data == "outside":
        # pre-training (spec stage 7): the version's public sources only, one class, "urchin"
        if not mtype.get("outside"):
            raise ValueError(f"{mtype['name']} cannot be trained on outside data yet.")
        if not manifest.get("outside"):
            raise ValueError(f"Version {manifest['number']} holds no outside data: add a source "
                             "on the Datasets page and freeze a new version.")
        class_ids = []
    d = mtype["defaults"]
    try:
        epochs = int(form.get("epochs") or d["epochs"])
        input_px = int(form.get("input") or d["input"])
        seed = int(form.get("seed") or d["seed"])
        val_every = int(form.get("val_every") or d.get("val_every", 1))
    except ValueError:
        raise ValueError("Epochs, input size, seed and \"validate every\" must be whole numbers.") from None
    if not 1 <= val_every <= 1000:
        raise ValueError("Validate every: between 1 and 1000 epochs.")
    augment = form.get("augment") or d["augment"]
    if not 1 <= epochs <= 1000:
        raise ValueError("Epochs must be between 1 and 1000.")
    if input_px not in mtype["inputs"]:
        raise ValueError(f"Input size must be one of {mtype['inputs']} for {mtype['name']}.")
    if augment not in mtype["presets"]:
        raise ValueError(f"Unknown augmentation preset {augment!r}.")
    names = {c["id"]: c["name"] for c in manifest["classes"]}
    class_names = [outside.CLASS_NAME] if data == "outside" else [names[i] for i in class_ids]
    if mtype.get("modeldef"):                 # a real model type: S6's arch and arch key (D53, D54)
        import importlib

        from app import modeldefs
        arch = importlib.import_module(mtype["modeldef"]).arch(
            mtype["size"], input_px, class_names)
        key = modeldefs.arch_key(arch)
    else:
        arch = {"kind": mtype["kind"], "family": mtype["family"], "size": mtype["size"],
                "variant": mtype.get("variant") or "", "input": [input_px, input_px],
                "classes": class_names, "licence": mtype.get("licence", "")}
        key = arch_key(mtype, input_px, len(class_names))
    train = {"model_type": mtype_key, "trainer": mtype["trainer"], "data": data,
             # "outside" = what the version holds; what a run really trained on is in
             # recorded[].outside_used, written by the trainer
             "dataset": {"version": manifest["number"], "hash": manifest["hash"],
                         "outside": [{"key": o["key"], "licence": o["licence"]}
                                     for o in manifest.get("outside", [])]},
             "class_ids": class_ids, "tile": input_px, "epochs": epochs, "augment": augment,
             "seed": seed, "start": start, "val_every": val_every}
    return key, {"arch": arch, "train": train}
