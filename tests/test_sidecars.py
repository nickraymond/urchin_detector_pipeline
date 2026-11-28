"""macOS "._" sidecars on the exFAT data disk are skipped wherever data_dir is listed (2026-11-27, job 8)."""

import re
from pathlib import Path

from app import sidecars
from app.fit import to_int8, to_int8_rfdetr
from app.sam import results
from app.stages import label

APP = Path(__file__).resolve().parents[1] / "app"


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_real_drops_only_sidecars():
    names = ["model.tflite", "._model.tflite", ".hidden", "a._b"]
    assert [p.name for p in sidecars.real(Path(n) for n in names)] == ["model.tflite", ".hidden", "a._b"]


def test_int8_step_finds_one_quantised_file_beside_its_sidecar(tmp_path):
    tf = tmp_path / "tf"
    real = touch(tf / "model_sim_full_integer_quant.tflite")
    touch(tf / "._model_sim_full_integer_quant.tflite", b"\0" * 4096)       # what macOS writes on exFAT
    touch(tf / "model_sim_float32.tflite")
    assert to_int8.quantised_files(tf) == [real]


def test_rfdetr_step_lists_no_sidecars(tmp_path):
    tf = tmp_path / "tf"
    touch(tf / "model.onnx", b"x" * 100)
    touch(tf / "._model.onnx", b"x" * 4096)               # larger than the model: must not be the one copied
    touch(tf / "inference_model_float32.tflite")
    touch(tf / "._inference_model_float32.tflite")
    assert [p.name for p in to_int8_rfdetr.tf_files(tf, "*.onnx")] == ["model.onnx"]
    assert [p.name for p in to_int8_rfdetr.tf_files(tf, "*.tflite")] == ["inference_model_float32.tflite"]


def test_sam_renames_and_labeled_list_skip_sidecars(tmp_path):
    raw = results.raw_dir(tmp_path)
    touch(raw / "._0001.jpg.json", b"\0\x05binary")      # AppleDouble bytes, not JSON
    touch(raw / "0001.jpg.json", b"{}")
    assert results.follow_renames(tmp_path, []) == 0      # does not try to read the sidecar
    assert [f for f, _ in label.labeled_frames(tmp_path, ["0001.jpg", "._0001.jpg"])] == ["0001.jpg"]


GLOB = re.compile(r"""\.r?glob\(f?["']([^"']*)["']\)""")


def test_every_wildcard_glob_in_app_skips_sidecars():
    """A glob whose last part starts with "*" matches "._" files too, so it must go through sidecars.real.
    (A pattern with a fixed start, such as "val_*.png", cannot match a "._" name.)"""
    bad = []
    for path in APP.rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            for pattern in GLOB.findall(line):
                if pattern.split("/")[-1].startswith("*") and "sidecars.real(" not in line:
                    bad.append(f"{path.relative_to(APP.parent)}:{n}: {line.strip()}")
    assert not bad, "globs that would match macOS ._ files:\n" + "\n".join(bad)
