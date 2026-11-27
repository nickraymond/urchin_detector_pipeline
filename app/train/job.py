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


# ---- model types the Train page offers (bite 5); bite 4b adds YOLOX here ----
# key -> {"name", "kind", "trainer" (a key of TRAINERS), "family", "size", "variant" (str),
#         "inputs": [px, ...], "presets": [augmentation presets], "defaults": {"epochs", "input",
#         "augment", "seed"}, "why": {field: one short sentence}, "licence"}
MODEL_TYPES: dict[str, dict] = {}


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
    d = mtype["defaults"]
    try:
        epochs = int(form.get("epochs") or d["epochs"])
        input_px = int(form.get("input") or d["input"])
        seed = int(form.get("seed") or d["seed"])
    except ValueError:
        raise ValueError("Epochs, input size and seed must be whole numbers.") from None
    augment = form.get("augment") or d["augment"]
    if not 1 <= epochs <= 1000:
        raise ValueError("Epochs must be between 1 and 1000.")
    if input_px not in mtype["inputs"]:
        raise ValueError(f"Input size must be one of {mtype['inputs']} for {mtype['name']}.")
    if augment not in mtype["presets"]:
        raise ValueError(f"Unknown augmentation preset {augment!r}.")
    names = {c["id"]: c["name"] for c in manifest["classes"]}
    arch = {"kind": mtype["kind"], "family": mtype["family"], "size": mtype["size"],
            "variant": mtype.get("variant") or "", "input": [input_px, input_px],
            "classes": [names[i] for i in class_ids], "licence": mtype.get("licence", "")}
    train = {"model_type": mtype_key, "trainer": mtype["trainer"],
             "dataset": {"version": manifest["number"], "hash": manifest["hash"],
                         "outside": [{"key": o["key"], "licence": o["licence"]}
                                     for o in manifest.get("outside", [])]},
             "class_ids": class_ids, "tile": input_px, "epochs": epochs, "augment": augment,
             "seed": seed, "start": start}
    return arch_key(mtype, input_px, len(class_ids)), {"arch": arch, "train": train}
