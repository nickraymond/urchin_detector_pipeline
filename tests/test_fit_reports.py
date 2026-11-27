"""S6 bite 3b: the report parsers, on small excerpts of real compiler output (tests/fixtures/fit/: the
known-good YOLOX-Nano run of 2026-11-27, the earlier NanoDet gate run, and the RF-DETR Nano first look),
and the two compiler modules' commands and missing-tool messages. No compiler runs here."""

from pathlib import Path

import pytest

from app.fit import ae3, convert, n6, pi, reports
from app.fit.convert_worker import ToolsMissing

FIX = Path(__file__).parent / "fixtures" / "fit"


def text(name: str) -> str:
    return (FIX / name).read_text()


def test_vela_summary_and_log_of_the_known_good_model():
    s = reports.vela_summary(text("vela_yolox_nano_summary.csv"))
    assert s == {"estimate_ms": 28.08, "memory_bytes": 524672, "flash_bytes": 964800,
                 "accelerator": "Ethos_U55_256", "system_config": "RTSS_HP_DTCM_MRAM"}
    assert reports.vela_log(text("vela_yolox_nano_log.txt")) == {"npu": 378, "cpu": 0, "failing_layers": []}


def test_vela_log_names_the_operators_left_on_the_cpu():
    placed = reports.vela_log(text("vela_rfdetr_nano_excerpt_log.txt"))
    assert (placed["npu"], placed["cpu"]) == (3789, 293)
    first = placed["failing_layers"][0]
    assert first == {"name": "model_543/tf.compat.v1.gather_nd_2/GatherNd", "type": "GATHER_ND",
                     "why": "Unsupported opType"}
    second = placed["failing_layers"][1]
    assert second["type"] == "CONCATENATION" and "Int64" in second["why"] and "64-bit" in second["why"]
    assert len(placed["failing_layers"]) >= 8 and all(l["why"] for l in placed["failing_layers"])


def test_st_report_counts_and_hybrid_steps():
    r = reports.stedgeai_report(text("stedgeai_yolox_nano_epochs.txt"))
    assert (r["npu"], r["hybrid"], r["cpu"], r["total"]) == (116, 2, 0, 118)
    assert r["failing_layers"] == [
        {"name": "epoch_63", "type": "DepthToSpace", "why": "runs partly on the CPU (hybrid)"},
        {"name": "epoch_74", "type": "DepthToSpace", "why": "runs partly on the CPU (hybrid)"}]
    n = reports.stedgeai_report(text("stedgeai_nanodet_epochs.txt"))
    assert (n["npu"], n["hybrid"], n["cpu"], n["total"]) == (158, 36, 2, 196)
    kinds = {l["type"] for l in n["failing_layers"]}
    assert "Transpose" in kinds and len(n["failing_layers"]) == 38
    assert sum(l["why"].startswith("runs on the CPU") for l in n["failing_layers"]) == 2
    with pytest.raises(ValueError, match="not an ST Edge AI Core"):
        reports.stedgeai_report("something else")


def test_st_sizes_and_error_lines():
    assert reports.stedgeai_sizes(text("n6_compile_yolox_nano_log.txt")) == {
        "file_bytes": 1166008, "params_bytes": 911760, "memory_bytes": 573440}
    err = reports.error_lines(text("n6_compile_rfdetr_nano_fail_log.txt"))
    assert "INTERNAL ERROR: 'NoneType' object is not subscriptable" in err and "\x1b" not in err
    assert reports.error_lines("a\nb\nc\n", keep=2) == "b\nc"
    tool = "noise\nTOOL ERROR: matmul: Input operand 1 has a mismatch in its core dimension 0\nmore noise\n"
    assert reports.error_lines(tool) == "TOOL ERROR: matmul: Input operand 1 has a mismatch in its core dimension 0"


def job(tmp_path, **fit):
    return {"tools_dir": str(tmp_path / "tools"), "fit": convert.fit_settings(fit)}


def test_fit_settings_defaults_and_overrides(tmp_path):
    s = convert.fit_settings(None)
    assert s["docker_image"] == "firmware-builder:latest" and "~" not in s["sdk_dir"] + s["openmv_dir"]
    assert convert.fit_settings({"sdk_dir": "/x", "other": 1}) == {**s, "sdk_dir": "/x"}
    assert convert.COMPILERS == {"n6": "app.fit.n6", "ae3": "app.fit.ae3", "pi": "app.fit.pi"}


def test_n6_command_mounts_the_toolchain_read_only(tmp_path):
    cmd = n6.docker_command(tmp_path / "int8" / "model_int8.tflite", tmp_path / "out",
                            convert.fit_settings({"sdk_dir": "/sdk-here", "openmv_dir": "/omv-here"}))
    assert "/sdk-here:/sdk:ro" in cmd and "/omv-here:/omv:ro" in cmd and f"{tmp_path / 'int8'}:/in:ro" in cmd
    assert f"{tmp_path / 'out'}:/out" in cmd and "--target stm32n6" in cmd[-1] and "/in/model_int8.tflite" in cmd[-1]
    assert not any(word in " ".join(cmd) for word in ("dfu", "ssh", "--privileged", "/dev"))   # never hardware
    assert cmd[cmd.index("--network") + 1] == "none"


def test_missing_tools_are_a_job_error_with_a_fix(tmp_path):
    with pytest.raises(ToolsMissing, match="Arm Vela not found"):
        ae3.compile(tmp_path / "m.tflite", tmp_path, job(tmp_path))
    with pytest.raises(ToolsMissing, match=r"OpenMV SDK not found.*\[fit\] sdk_dir"):
        n6.compile(tmp_path / "m.tflite", tmp_path, job(tmp_path, sdk_dir=str(tmp_path / "none")))


def fake_tools(tmp_path, script: str) -> dict:
    """A stand-in for <tools_dir>/convert-venv/bin/python that prints what a test wants."""
    py = tmp_path / "tools" / "convert-venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n" + script + "\n")
    py.chmod(0o755)
    return job(tmp_path)


def test_pi_step_reports_cpu_only_and_laptop_figures_as_laptop_figures(tmp_path):
    model = tmp_path / "model_int8.tflite"
    model.write_bytes(b"TFL3" * 10)
    ran = ('{"runtime": "LiteRT (ai-edge-litert) 2.2.0", "operators": 5, "operator_types": {"CONV_2D": 4, '
           '"FlexErf": 1}, "not_builtin": ["FlexErf"], "input": {"shape": [1, 8, 8, 3], "dtype": "int8"}, '
           '"outputs": [], "laptop_ms": 1.5, "runs": 20, "laptop_peak_bytes": 1000}')
    out = tmp_path / "out"
    out.mkdir()
    fit = pi.compile(model, out, fake_tools(tmp_path, f"echo noise; echo '{ran}'"))
    assert fit["converts"] and fit["placement"] == {"npu": 0, "hybrid": 0, "cpu": 5, "total": 5, "unit": "operators"}
    assert fit["file_bytes"] == 40 and (out / "model_int8.tflite").is_file()
    assert fit["estimate_ms"] is None and fit["memory_bytes"] is None        # nothing claimed for the Pi
    assert fit["laptop"]["ms"] == 1.5 and "nothing about the Pi" in fit["laptop"]["note"]
    assert fit["memory_limit_bytes"] == 512 * 1024 * 1024 and fit["limit_bytes"] is None
    assert [l["name"] for l in fit["failing_layers"]] == ["FlexErf"]


def test_pi_step_failure_is_a_result_and_a_missing_environment_is_a_job_error(tmp_path):
    model = tmp_path / "model_int8.tflite"
    model.write_bytes(b"x")
    with pytest.raises(ToolsMissing, match="Tool environment not found"):
        pi.compile(model, tmp_path, job(tmp_path))
    fit = pi.compile(model, tmp_path, fake_tools(tmp_path, "echo 'ValueError: Model provided has model identifier' >&2; exit 1"))
    assert fit["converts"] is False and "model identifier" in fit["error"] and "exit code 1" in fit["error"]
    fit = pi.compile(model, tmp_path, fake_tools(tmp_path.parent / (tmp_path.name + "_abort"), "kill -ABRT $$"))
    assert fit["converts"] is False and "aborted while preparing or running the model" in fit["error"]

