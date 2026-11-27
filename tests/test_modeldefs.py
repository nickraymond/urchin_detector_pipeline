"""app/modeldefs (D51, D54): candidates as data, folder names, checks. No model is built here unless
YOLOX's code and its imports are on this machine (the conversion environment); CI-style runs skip that."""

import ast
import importlib.util
from pathlib import Path

import pytest

from app import modeldefs
from app.modeldefs import yolox

PKG = Path(modeldefs.__file__).parent


def test_modeldefs_imports_nothing_else_from_app():
    for path in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith("app") or name.startswith("app.modeldefs"), (path.name, name)


def test_init_does_not_import_torch_at_module_level():
    tree = ast.parse((PKG / "__init__.py").read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in ast.dump(n) for n in top)


def test_detector_candidates_are_complete_recipes():
    cands = modeldefs.candidates("detector")
    keys = [modeldefs.arch_key(a) for a in cands]
    assert "yolox-nano-conv-256-c5" in keys and "yolox-tiny-conv-256-c5" in keys
    assert len(set(keys)) == len(keys)
    for a in cands:
        assert modeldefs.check(a) == []
        assert a["classes"] == modeldefs.BUILD2_CLASSES
        assert a["licence"] in ("Apache-2.0", "MIT", "BSD-3-Clause") and a["licence_mark"] is None
        assert len(a["code"]["commit"]) == 40
    assert modeldefs.candidates("segmenter") == []   # bite 2b


def test_arch_key_shapes():
    a = yolox.arch("nano", 256, ["x"])
    assert modeldefs.arch_key(a) == "yolox-nano-conv-256-c1"
    a["input"] = [192, 256]
    assert modeldefs.arch_key(a) == "yolox-nano-conv-192x256-c1"
    a["family"] = "Bad Name"
    with pytest.raises(modeldefs.ModelDefError):
        modeldefs.arch_key(a)


@pytest.mark.parametrize("change, word", [
    ({"kind": "classifier"}, "kind"),
    ({"family": "yolov8"}, "not in the app"),
    ({"input": [250, 256]}, "multiples of 32"),
    ({"input": 256}, "height, width"),
    ({"classes": []}, "no classes"),
    ({"size": "huge"}, "size"),
    ({"variant": {"stem": "focus"}}, "plain first layer"),
    ({"code": {"repo": "x"}}, "repo and commit"),
])
def test_check_says_what_is_wrong(change, word):
    a = {**yolox.arch("nano", 256, ["x"]), **change}
    problems = modeldefs.check(a)
    assert problems and any(word in p for p in problems), problems
    with pytest.raises(modeldefs.ModelDefError):
        modeldefs.build(a)


def test_check_names_a_missing_field():
    a = yolox.arch("nano", 256, ["x"])
    del a["licence"]
    assert modeldefs.check(a) == ["the recipe has no licence"]


def test_missing_checkout_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("URCHIN_TOOLS_DIR", str(tmp_path))
    with pytest.raises(modeldefs.ModelDefError, match="set the tools up"):
        modeldefs.build(yolox.arch("nano", 256, ["x"]))


def _yolox_here() -> bool:
    return ((modeldefs.tools_dir() / "third_party" / "YOLOX").is_dir()
            and all(importlib.util.find_spec(m) for m in ("torch", "loguru", "tabulate", "cv2")))


@pytest.mark.skipif(not _yolox_here(), reason="YOLOX's code or its imports are not in this environment")
@pytest.mark.parametrize("size, px", [("nano", 256), ("tiny", 320)])
def test_yolox_builds_untrained_with_raw_maps(size, px):
    import torch
    a = yolox.arch(size, px, modeldefs.BUILD2_CLASSES)
    outs = modeldefs.build(a)(torch.zeros(1, 3, px, px))
    assert [tuple(o.shape) for o in outs] == [(1, 10, px // s, px // s) for s in yolox.STRIDES]
    assert all(0 <= float(o[:, 4:].min()) and float(o[:, 4:].max()) <= 1 for o in outs)
