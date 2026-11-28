"""A stored fit result knows what made its 8-bit file; one from an older recipe says "Run again" (QE PR 92 nit 1)."""

import json
import sys
from pathlib import Path

import pytest

from app import modeldefs
from app.fit import convert, table
from app.jobctl import JobControl
from app.modeldefs import yolox
from tests.test_convert import wait_state
from tests.test_fit_page import site, write_fit  # noqa: F401  (site is a fixture)


def arch_of(key: str) -> dict:
    return next(a for a in modeldefs.candidates() if modeldefs.arch_key(a) == key)


def test_made_with_is_the_export_settings_and_the_code_pin():
    arch = arch_of("rfdetr-nano-384-c5")
    made = convert.made_with(arch)
    assert made["export"]["route"] == "rfdetr-own" and made["code"] == json.loads(json.dumps(arch["code"]))


def test_older_recipe_compares_the_record_and_falls_back_to_int8_json():
    arch = arch_of("rfdetr-nano-384-c5")
    now = convert.made_with(arch)
    assert not convert.older_recipe(None, arch) and not convert.older_recipe({"made_with": now}, arch)
    old_pin = {**now, "code": {**now["code"], "commit": "pip:rfdetr==1.9.4"}}
    assert convert.older_recipe({"made_with": old_pin}, arch)
    old_route = {**now["export"], "route": "onnx2tf", "env": "rfdetr-venv"}     # a result from before bite 2e
    assert convert.older_recipe({"int8": {"export": old_route}}, arch)
    assert not convert.older_recipe({"int8": {"export": now["export"]}}, arch)
    assert not convert.older_recipe({"converts": False, "stage": "8-bit model"}, arch)   # no record: not marked


def test_the_job_records_what_made_the_8_bit_file_in_every_camera_result(tmp_path):
    jc = JobControl(tmp_path / "jobs")
    jc.register(convert.convert_kind())
    try:
        arch = yolox.arch("nano", 256, modeldefs.BUILD2_CLASSES)
        p = {**convert.params(arch, "fakecam", tmp_path, tmp_path / "tools"), "compiler": "tests.fake_compiler",
             "tool_python": sys.executable, "int8_module": "tests.fake_int8"}
        assert wait_state(jc, jc.submit("convert", p), "done", "error")["state"] == "done"
        folder = Path(p["folder"])
        fit = convert.read_fit(folder, "fakecam")
        assert fit["made_with"] == convert.made_with(arch) and not convert.older_recipe(fit, arch)
        assert json.loads(convert.made_with_path(folder).read_text()) == fit["made_with"]
    finally:
        jc.close()


def test_page_marks_a_result_made_with_an_older_recipe(site):  # noqa: F811
    client, data = site
    arch = arch_of("yolox-nano-conv-256-c5")
    old = {**convert.made_with(arch), "export": {**convert.made_with(arch)["export"], "opset": 11}}
    write_fit(data, "yolox-nano-conv-256-c5", "ae3", converts=True, file_bytes=1, limit_bytes=1 << 24,
              placement={"npu": 9, "hybrid": 0, "cpu": 0, "total": 9, "unit": "operators"}, made_with=old)
    write_fit(data, "yolox-nano-conv-256-c5", "n6", converts=True, file_bytes=1, limit_bytes=1 << 24,
              placement={"npu": 9, "hybrid": 0, "cpu": 0, "total": 9, "unit": "epochs"},
              made_with=convert.made_with(arch))
    cells = {r["kind"]: r for r in table.table(data)}["detector"]["cells"]
    assert cells["ae3"]["best"]["older"] and not cells["n6"]["best"]["older"]
    page = client.get("/fit").text
    assert page.count("made with an older recipe: Run again") == 1 and "; <b>made with an older recipe</b>" in page
