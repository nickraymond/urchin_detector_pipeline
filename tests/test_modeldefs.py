"""app/modeldefs (D51, D54): candidates as data, folder names, checks. No model is built here unless
YOLOX's code and its imports are on this machine (the conversion environment); CI-style runs skip that."""

import ast
import importlib.util
from pathlib import Path

import pytest

from app import modeldefs
from app.modeldefs import yolox

PKG = Path(modeldefs.__file__).parent
PERMISSIVE = ("Apache-2.0", "MIT", "BSD-3-Clause", "own code")


def _yolox_here() -> bool:
    return ((modeldefs.tools_dir() / "third_party" / "YOLOX").is_dir()
            and all(importlib.util.find_spec(m) for m in ("torch", "loguru", "tabulate", "cv2")))


def test_modeldefs_imports_nothing_else_from_app():
    for path in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith("app") or name.startswith("app.modeldefs"), (path.name, name)


def test_no_file_imports_torch_or_model_code_at_module_level():
    """candidates() imports every family's file; none may load torch or third-party model code."""
    for path in PKG.glob("*.py"):
        top = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        for n in top:
            assert not any(word in ast.dump(n) for word in ("torch", "yolox.", "rfdetr'")), (path.name, ast.dump(n))


def test_detector_candidates_are_complete_recipes():
    cands = modeldefs.candidates("detector")
    keys = [modeldefs.arch_key(a) for a in cands]
    assert "yolox-nano-conv-256-c5" in keys and "yolox-tiny-conv-256-c5" in keys
    assert len(set(keys)) == len(keys)
    for a in cands:
        assert modeldefs.check(a) == []
        assert a["classes"] == modeldefs.BUILD2_CLASSES
        assert a["licence"] in PERMISSIVE and a["licence_mark"] is None
        assert len(a["code"]["commit"]) == 40 or a["code"]["commit"].startswith("pip:")
        assert modeldefs.export_settings(a)["env"] in ("convert-venv", "rfdetr-venv")
    assert {"rfdetr-nano-384-c5", "rfdetr-nano-256-c5"} <= set(keys)
    keys = [modeldefs.arch_key(a) for a in modeldefs.candidates("instance")]
    assert keys == ["rfdetr-seg-nano-312-c5", "yoloxproto-nano-conv-256-c5", "yoloxproto-tiny-conv-256-c5"]
    for a in modeldefs.candidates("instance"):
        assert modeldefs.check(a) == [] and a["licence"] in PERMISSIVE and a["licence_mark"] is None


def test_instance_candidate_checks():
    from app.modeldefs import rfdetr, yoloxproto
    seg = rfdetr.seg_arch(312, ["x"])
    assert seg["outputs"]["names"] == ["dets", "labels", "masks"]
    assert any("multiple of 12" in p for p in modeldefs.check({**seg, "input": [320, 320]}))
    assert any("only RF-DETR Nano" in p for p in modeldefs.check({**seg, "kind": "detector"}))
    assert any("only RF-DETR Nano" in p for p in modeldefs.check({**rfdetr.arch(384, ["x"]), "kind": "instance"}))
    proto = yoloxproto.arch("nano", 256, ["x"])
    assert proto["outputs"]["names"][-1] == "protos"
    assert any("multiples of 32" in p for p in modeldefs.check({**proto, "input": [250, 250]}))


@pytest.mark.skipif(not _yolox_here(), reason="YOLOX's code or its imports are not in this environment")
def test_yolox_proto_builds_with_coefficients_and_prototypes():
    import torch
    from app.modeldefs import yoloxproto
    a = yoloxproto.arch("nano", 256, modeldefs.BUILD2_CLASSES)
    *maps, protos = modeldefs.build(a)(torch.zeros(1, 3, 256, 256))
    assert [tuple(o.shape) for o in maps] == [(1, 10 + yoloxproto.PROTOS, 256 // s, 256 // s) for s in (8, 16, 32)]
    assert tuple(protos.shape) == (1, yoloxproto.PROTOS, 64, 64)


def test_segmenter_candidates_build_untrained_and_give_one_scores_map():
    """torchvision and the plain net need only torch + torchvision, which the app's environment has."""
    import torch
    cands = modeldefs.candidates("segmenter")
    keys = [modeldefs.arch_key(a) for a in cands]
    assert keys == ["tvseg-lraspp-mnv3l-256-c6", "tvseg-deeplabv3-mnv3l-256-c6", "plainseg-s-256-c6", "plainseg-m-256-c6"]
    for a in cands:
        assert modeldefs.check(a) == [] and a["classes"][-1] == "other" and a["licence"] in PERMISSIVE
        assert ("own untrained design" in a.get("note", "")) == (a["family"] == "plainseg")
        model = modeldefs.build(a)
        assert not model.training
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 256, 256))
        side = 256 // a["outputs"].get("stride", 1)
        assert tuple(out.shape) == (1, 6, side, side), modeldefs.arch_key(a)


def test_plain_segmenter_uses_only_the_layer_types_it_promises():
    from torch import nn
    from app.modeldefs import plainseg
    allowed = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6, nn.Upsample, nn.Sequential)
    model = plainseg.build(plainseg.arch("s", 256, modeldefs.SEGMENTER_CLASSES))
    odd = {type(m).__name__ for m in model.modules() if m is not model and not isinstance(m, allowed)}
    assert odd == set()


def test_segmenter_checks():
    from app.modeldefs import plainseg, tvseg
    assert any("multiples of 32" in p for p in modeldefs.check(tvseg.arch("lraspp-mnv3l", 250, ["a", "other"])))
    assert any("one of" in p for p in modeldefs.check(tvseg.arch("fcn", 256, ["a", "other"])))
    assert any("multiples of 16" in p for p in modeldefs.check(plainseg.arch("s", 250, ["a", "other"])))
    assert any("one of" in p for p in modeldefs.check(plainseg.arch("xl", 256, ["a", "other"])))


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
    ({"classes": 5}, "list of class names"),
    ({"classes": "abc"}, "list of class names"),
    ({"classes": ["x", 3]}, "list of class names"),
    ({"code": "x"}, "table of names"),
    ({"variant": "conv"}, "table of names"),
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


def test_rfdetr_checks_and_export_settings():
    from app.modeldefs import rfdetr
    a = rfdetr.arch(384, ["x"])
    assert modeldefs.check(a) == [] and modeldefs.export_settings(a) == {"env": "rfdetr-venv", "opset": 17, "simplify": True}
    assert modeldefs.export_settings({}) == {"env": "convert-venv", "opset": 13, "simplify": False}
    assert any("multiple of 32" in p for p in modeldefs.check({**a, "input": [336, 336]}))
    assert any("multiple of 32" in p for p in modeldefs.check({**a, "input": [320, 384]}))
    assert any("Roboflow's own" in p for p in modeldefs.check({**a, "size": "pico"}))


@pytest.mark.skipif(importlib.util.find_spec("rfdetr") is None, reason="rfdetr is only in its tool environment")
def test_rfdetr_nano_builds_untrained_offline():
    import torch
    from app.modeldefs import rfdetr
    dets, labels = [o for o in modeldefs.build(rfdetr.arch(256, modeldefs.BUILD2_CLASSES))(torch.zeros(1, 3, 256, 256))
                    if o is not None]
    assert tuple(dets.shape) == (1, 300, 4) and tuple(labels.shape) == (1, 300, 6)


def test_rfdetr_missing_package_says_how_to_fix():
    if importlib.util.find_spec("rfdetr") is not None:
        pytest.skip("rfdetr is installed here")
    from app.modeldefs import rfdetr
    with pytest.raises(modeldefs.ModelDefError, match="rfdetr-venv"):
        modeldefs.build(rfdetr.arch(384, ["x"]))


def test_check_names_a_missing_field():
    a = yolox.arch("nano", 256, ["x"])
    del a["licence"]
    assert modeldefs.check(a) == ["the recipe has no licence"]


def test_missing_checkout_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("URCHIN_TOOLS_DIR", str(tmp_path))
    with pytest.raises(modeldefs.ModelDefError, match="set the tools up"):
        modeldefs.build(yolox.arch("nano", 256, ["x"]))


def test_third_party_loads_one_package_and_leaves_sys_path_alone(tmp_path, monkeypatch):
    import sys
    root = tmp_path / "third_party" / "Thing"
    (root / "fakepkg").mkdir(parents=True)
    (root / "fakepkg" / "__init__.py").write_text("from fakepkg.sub import VALUE\n")
    (root / "fakepkg" / "sub.py").write_text("VALUE = 7\n")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("SHADOW = True\n")
    monkeypatch.setenv("URCHIN_TOOLS_DIR", str(tmp_path))
    before = list(sys.path)
    try:
        assert modeldefs.third_party("Thing", "a" * 40, "repo", package="fakepkg") == root
        import fakepkg
        assert fakepkg.VALUE == 7 and sys.path == before
        assert not any(str(root) == p for p in sys.path)
    finally:
        sys.modules.pop("fakepkg", None)
        sys.modules.pop("fakepkg.sub", None)


@pytest.mark.skipif(not _yolox_here(), reason="YOLOX's code or its imports are not in this environment")
@pytest.mark.parametrize("size, px", [("nano", 256), ("tiny", 320)])
def test_yolox_builds_untrained_with_raw_maps(size, px):
    import torch
    a = yolox.arch(size, px, modeldefs.BUILD2_CLASSES)
    outs = modeldefs.build(a)(torch.zeros(1, 3, px, px))
    assert [tuple(o.shape) for o in outs] == [(1, 10, px // s, px // s) for s in yolox.STRIDES]
    assert all(0 <= float(o[:, 4:].min()) and float(o[:, 4:].max()) <= 1 for o in outs)
