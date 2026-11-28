"""S7 bite 3: ONNX Runtime, the Pi's second runtime candidate (D85). The agent runs an ONNX file under it; a Pi
card's runtime follows its model files (.tflite: LiteRT, .onnx: ONNX Runtime); the install is its own pinned
environment (D89). The runtime test uses the conversion environment's onnxruntime 1.29.0 when this machine has it."""

import re
import subprocess
from pathlib import Path

import pytest

from app.bench import card, units
from app.bench.agent.bench_agent import INSTALLS, SUFFIX_RUNTIME
from app.bench.board import Model, open_board
from app.bench.pi_board import PiBoard
from app.bench.transport import AGENT_FILES, LocalTransport
from app.bench.worker import board_settings
from tests.test_bench_cards import form

CONVERT_PY = Path("~/NereusData/urchin_detector_pipeline/tools/convert-venv/bin/python").expanduser()
TINY = """
import sys, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
w = numpy_helper.from_array(np.ones((4, 3, 3, 3), np.float32) / 27, "w")
g = helper.make_graph([helper.make_node("Conv", ["images", "w"], ["out"], pads=[1, 1, 1, 1])], "tiny",
                      [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 16, 16])],
                      [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)], [w])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), sys.argv[1])
"""


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    for arch in ("yolox_nano", "yolox_tiny"):
        for name in ("model_int8.tflite", "model_qdq.onnx"):
            (d / "fit" / arch / "targets" / "pi").mkdir(parents=True, exist_ok=True)
            (d / "fit" / arch / "targets" / "pi" / name).write_bytes(b"model")
    return d


def test_a_pi_cards_runtime_follows_its_model_files(data):
    b = units.Bench()
    assert card.build(data, b, form(board="pi", models="fit/yolox_nano/targets/pi/model_int8.tflite"))["runtime"] == "litert"
    assert card.build(data, b, form(board="pi", models="fit/yolox_nano/targets/pi/model_qdq.onnx"))["runtime"] == "onnxruntime"
    with pytest.raises(card.CardError, match="all .tflite or all .onnx"):
        card.build(data, b, form(board="pi", models="fit/yolox_nano/targets/pi/model_int8.tflite\n"
                                                    "fit/yolox_tiny/targets/pi/model_qdq.onnx"))
    (data / "odd.pt").write_bytes(b"x")
    with pytest.raises(card.CardError, match=r"must end in \.tflite or \.onnx"):
        card.build(data, b, form(board="pi", models="odd.pt"))
    assert "runtime" not in card.build(data, b, form(board="n6", models="fit/yolox_nano/targets/pi/model_int8.tflite"))


def test_a_real_pi_gets_the_cards_runtime_and_its_own_environment(tmp_path):
    bench = units.parse({"field_units": ["field-pi"], "units": {"bench-a": {"ssh_host": "bench-a", "cameras": []}}})
    unit = bench.unit("bench-a")
    s = board_settings({"board": "pi", "threads": 2, "runtime": "onnxruntime"}, unit)
    assert s == {"threads": 2, "runtime": "onnxruntime"}
    assert board_settings({"board": "pi"}, unit) == {"threads": 1, "runtime": "litert"}    # cards from before bite 3c
    board = open_board(unit, "pi", tmp_path, bench=bench, **s)
    assert isinstance(board, PiBoard) and board.install == "onnxruntime" and board.transport.runtime == "onnxruntime"
    assert board.transport.argv()[-2:] == ["--runtime", "onnxruntime"]                  # built; nothing connected


def test_the_pin_file_is_hash_pinned_and_goes_with_the_agent():
    spec = INSTALLS["onnxruntime"]
    f = Path(__file__).resolve().parents[1] / "app" / "bench" / "agent" / spec["requirements"]
    pins = [l for l in f.read_text().splitlines() if l and not l.startswith("#")]
    assert {l.split("==")[0] for l in pins} == {"onnxruntime", "flatbuffers", "numpy", "packaging", "protobuf"}
    assert all(re.fullmatch(r"[a-z0-9-]+==[0-9.]+ --hash=sha256:[0-9a-f]{64}", l) for l in pins)
    assert f in AGENT_FILES and SUFFIX_RUNTIME[".onnx"] == "onnxruntime"


@pytest.mark.skipif(not CONVERT_PY.is_file(), reason="no conversion environment (onnxruntime 1.29.0) on this machine")
def test_the_agent_times_an_onnx_file_under_onnx_runtime(tmp_path):
    model = tmp_path / "tiny.onnx"
    subprocess.run([str(CONVERT_PY), "-c", TINY, str(model)], check=True)
    board = PiBoard(units.Bench().unit("fake"), tmp_path,
                    LocalTransport(tmp_path / "agent", runtime="onnxruntime", python=str(CONVERT_PY)), threads=2)
    try:
        loaded = board.load(Model("tiny", model))
        assert loaded["threads"] == 2 and loaded["io"]["tiny"]["input"] == {"shape": [1, 3, 16, 16], "dtype": "float32"}
        t = board.time("tiny", 20)
        assert t["n"] == 20 and t["threads"] == 2 and 0 < t["ms_min"] <= t["ms_mean"] <= t["ms_max"]
        assert board.info()["runtime"] == "onnxruntime 1.29.0"
    finally:
        board.close()


def test_without_a_float_onnx_file_the_pi_step_says_why_and_keeps_its_litert_result(tmp_path):
    from app.fit.pi import qdq
    r = qdq(tmp_path / "python", tmp_path / "model.onnx", tmp_path)
    assert r["file"] is None and "no model.onnx" in r["error"]


@pytest.mark.skipif(not CONVERT_PY.is_file(), reason="no conversion environment (onnxruntime 1.29.0) on this machine")
def test_the_pi_step_makes_the_qdq_file_and_the_agent_runs_it(tmp_path):
    """Lead call (a), lead 2: S6's Pi step writes targets/pi/model_qdq.onnx with D55's seeded calibration."""
    from app.fit.pi import QDQ_NAME, qdq
    src = tmp_path / "model.onnx"
    subprocess.run([str(CONVERT_PY), "-c", TINY, str(src)], check=True)
    out = tmp_path / "pi"
    out.mkdir()
    r = qdq(CONVERT_PY, src, out)
    assert r["file"] == QDQ_NAME and r["file_bytes"] == (out / QDQ_NAME).stat().st_size, r
    assert r["runtime"] == "onnxruntime 1.29.0" and r["operator_types"].get("QuantizeLinear", 0) >= 1
    assert "seeded noise" in r["calibration"] and not list(out.glob("*.prep.onnx"))
    board = PiBoard(units.Bench().unit("fake"), tmp_path,
                    LocalTransport(tmp_path / "agent", runtime="onnxruntime", python=str(CONVERT_PY)), threads=1)
    try:
        board.load(Model("tiny", out / QDQ_NAME))
        assert board.time("tiny", 10)["n"] == 10
    finally:
        board.close()
