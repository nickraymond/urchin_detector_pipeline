"""S7 bite 3: the bench agent, its JSON-line protocol, the transports and the Pi board. Fake transport only (the
agent as a child process on the laptop, "fake" runtime); no ssh is ever run, no unit is touched."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.bench import riglog
from app.bench.agent import bench_agent as ag
from app.bench.board import NO_ANSWER, NOT_LOADED, READBACK, BoardError, Model
from app.bench.pi_board import AGENT_UNDO, PiBoard
from app.bench.transport import LocalTransport, SshTransport, resolved_host
from app.bench.units import Bench, Refused, parse


def talk(agent, *reqs):
    out = []
    ag.serve(agent, [json.dumps(r) if isinstance(r, dict) else r for r in reqs], out.append)
    return out


def model_file(tmp_path, name="m.tflite", data=b"model bytes"):
    p = tmp_path / name
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def test_agent_answers_every_line_with_its_id_and_a_stop_rule_on_refusal(tmp_path):
    agent = ag.Agent(tmp_path / "agent", "fake")
    (tmp_path / "agent" / "models").mkdir(parents=True)
    out = talk(agent, {"id": 1, "op": "hello"}, "not json", "[1, 2]", {"id": 2, "op": "fly"},
               {"id": 3, "op": "info", "kind": "n6"}, {"id": 4, "op": "time", "name": "x", "n": 3},
               {"id": 5, "op": "load", "files": {"x": "0" * 64}})
    assert out[0]["ok"] and out[0]["agent"] == ag.VERSION and out[0]["runtime"] == "fake"
    assert [o["ok"] for o in out] == [True, False, False, False, False, False, False]
    assert out[1]["id"] is None and "JSON object" in out[2]["error"] and "unknown op" in out[3]["error"]
    assert out[4]["stop"] == ag.NOT_BUILT and "bites 4-5" in out[4]["error"]
    assert out[5]["stop"] == NOT_LOADED and out[6]["stop"] == "load_failed" and "did not arrive" in out[6]["error"]
    assert [o["id"] for o in out] == [1, None, None, 2, 3, 4, 5]


def test_agent_checks_the_copy_and_times_a_window(tmp_path):
    folder = tmp_path / "agent"
    agent = ag.Agent(folder, "fake")
    src, sha = model_file(tmp_path)
    (folder / "models").mkdir(parents=True)
    (folder / "models" / ag.file_name("fit/yolox/model.tflite")).write_bytes(src.read_bytes())
    bad, ok, timed = talk(agent, {"id": 1, "op": "load", "files": {"fit/yolox/model.tflite": "f" * 64}},
                          {"id": 2, "op": "load", "files": {"fit/yolox/model.tflite": sha}, "threads": 2},
                          {"id": 3, "op": "time", "name": "fit/yolox/model.tflite", "n": 5})
    assert bad["stop"] == READBACK and ok["loaded"] == ["fit/yolox/model.tflite"] and ok["threads"] == 2
    assert timed["n"] == 5 and timed["warmups"] == 2 and timed["threads"] == 2
    assert 0 <= timed["ms_min"] <= timed["ms_mean"] <= timed["ms_max"]


def test_file_names_and_proc_readouts(tmp_path):
    assert ag.file_name("fit/yolox-nano-conv-256-c5/int8/model_int8.tflite") == \
        "fit_yolox-nano-conv-256-c5_int8_model_int8.tflite"
    status = tmp_path / "status"
    status.write_text("Name:\tpython3\nVmHWM:\t   51200 kB\nVmRSS:\t 40000 kB\n")
    assert ag.proc_kib(status, "VmHWM") == 51200 * 1024 and ag.proc_kib(status, "Nope") is None
    assert ag.proc_kib(tmp_path / "no such file", "VmHWM") is None


def fake_unit():
    return Bench().unit("fake")


def test_pi_board_through_the_local_transport_logs_the_copy(tmp_path):
    data = tmp_path / "data"
    board = PiBoard(fake_unit(), data, LocalTransport(tmp_path / "agent"), threads=1)
    try:
        src, sha = model_file(tmp_path)
        info = board.info()
        assert info["kind"] == "pi" and info["agent"]["runtime"] == "fake" and info["model_area_bytes"] == 0
        loaded = board.load(Model("fit/yolox-nano-conv-256-c5", src))
        assert loaded["loaded"] == ["fit/yolox-nano-conv-256-c5"]
        w = board.time("fit/yolox-nano-conv-256-c5", 10)
        assert w["n"] == 10 and w["ms_mean"] >= 0
        board.reset()
        with pytest.raises(BoardError) as e:
            board.time("fit/yolox-nano-conv-256-c5", 10)
        assert e.value.stop == NOT_LOADED
        log = riglog.read(data)
        assert [r["state"] for r in log] == ["done"] and log[0]["fake"] is True
        assert log[0]["sha256"] == {"fit/yolox-nano-conv-256-c5": sha}
        board.install_agent()                                     # a local agent needs no copy: nothing logged
        assert len(riglog.read(data)) == 1
    finally:
        board.close()


def test_an_agent_that_does_not_answer_is_the_no_answer_stop(tmp_path):
    class Silent(LocalTransport):
        def argv(self):
            return [sys.executable, "-c", "import time; time.sleep(30)"]
    t = Silent(tmp_path, timeout_s=0.5)
    with pytest.raises(BoardError) as e:
        t.start()
    assert e.value.stop == NO_ANSWER and "no answer from the bench agent to 'hello'" in str(e.value)
    t.proc.kill()


class FakeRun:
    """Stands in for subprocess.run: records every command, prints a configuration for `ssh -G`."""

    def __init__(self, hostname="bench-pi.local", rc=0):
        self.calls, self.hostname, self.rc = [], hostname, rc

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        out = f"user pi\nhostname {self.hostname}\nport 22\n" if argv[:2] == ["ssh", "-G"] else ""
        return subprocess.CompletedProcess(argv, self.rc, out, "" if self.rc == 0 else "no such alias")


def test_ssh_alias_resolving_to_a_field_unit_is_refused_before_any_connection():
    bench = parse({"field_units": ["field-pi"], "units": {"benchpi": {"ssh_host": "benchpi", "cameras": ["n6"]}}})
    run = FakeRun(hostname="Field-Pi.local")
    t = SshTransport("benchpi", bench.field_units, run=run)
    with pytest.raises(Refused, match="field unit"):
        t.start()
    with pytest.raises(Refused, match="field unit"):
        t.put(Path("x"), "x")
    assert all(c[:2] == ["ssh", "-G"] for c in run.calls) and t.proc is None   # nothing but configuration reads
    with pytest.raises(Refused, match="could not read"):
        resolved_host("nope", FakeRun(rc=255))


def test_ssh_commands_for_a_bench_unit(tmp_path):
    run = FakeRun()
    t = SshTransport("benchpi", frozenset({"field-pi"}), run=run)
    assert t.check() == "bench-pi.local"
    assert t.argv()[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "benchpi"]
    assert t.argv()[6:] == ["python3", "urchin_bench/bench_agent.py", "urchin_bench", "--runtime", "litert"]
    t.install()
    t.put(tmp_path / "m.tflite", "fit/a/model.tflite")
    copies = [c for c in run.calls if c[0] == "scp"]
    assert copies[0][-1] == "benchpi:urchin_bench/bench_agent.py"
    assert copies[1][-1] == "benchpi:urchin_bench/models/fit_a_model.tflite"


def test_installing_the_agent_on_a_unit_is_a_logged_rig_change(tmp_path):
    from app.bench.units import Unit
    run = FakeRun()
    board = PiBoard(Unit("benchpi", "benchpi"), tmp_path, SshTransport("benchpi", frozenset(), run=run))
    board.install_agent()
    log = riglog.read(tmp_path)
    assert [r["state"] for r in log] == ["done"] and log[0]["undo"] == AGENT_UNDO and log[0]["fake"] is False
    assert riglog.undo_steps(log)[0]["what"] == "copy the bench agent"        # listed under "To put the rig back"


def test_a_window_longer_than_the_answer_limit_is_kept_alive(tmp_path):
    """QE PR 108 finding 1: 60 s is for silence, not for a long window (300 × 200 ms on the Pi Zero is 60 s)."""
    data = tmp_path / "data"
    slow = LocalTransport(tmp_path / "agent", timeout_s=1.0, extra=("--alive-s", "0.2", "--fake-ms", "50"))
    board = PiBoard(fake_unit(), data, slow)
    try:
        src, _ = model_file(tmp_path)
        board.load(Model("slow", src))
        w = board.time("slow", 60)                    # about 3 s of window against a 1 s answer limit
        assert w["n"] == 60 and w["ms_mean"] >= 50
    finally:
        board.close()
    silent = LocalTransport(tmp_path / "agent2", timeout_s=1.0, extra=("--alive-s", "30", "--fake-ms", "50"))
    board = PiBoard(fake_unit(), data, silent)
    try:
        board.load(Model("slow", src))
        with pytest.raises(BoardError) as e:          # no alive line within the limit: no_answer, as before
            board.time("slow", 60)
        assert e.value.stop == NO_ANSWER
    finally:
        board.close()


def test_the_pi_board_returns_every_field_the_bench_job_reads(tmp_path):
    """QE PR 108 finding 2: the same fields as FakeBoard.time (the bench job reads mem_after_load)."""
    from app.bench.board import FakeBoard
    src, _ = model_file(tmp_path)
    fake = FakeBoard(fake_unit(), "pi", tmp_path / "f")
    fake.load(Model("m", src))
    board = PiBoard(fake_unit(), tmp_path / "p", LocalTransport(tmp_path / "agent"))
    try:
        board.load(Model("m", src))
        assert set(fake.time("m", 3)) <= set(board.time("m", 3))
    finally:
        board.close()
