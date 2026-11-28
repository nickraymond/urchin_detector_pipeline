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
from app.bench.units import Bench, Refused, Unit, parse


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
        board.install_agent()                                     # the pin files into the local folder: logged
        assert [r["what"] for r in riglog.read(data)][0] == "copy the bench agent"
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
    t = SshTransport(bench.unit("benchpi"), bench.field_units, run=run)
    with pytest.raises(Refused, match="field unit"):
        t.start()
    with pytest.raises(Refused, match="field unit"):
        t.put(Path("x"), "x")
    assert all(c[:2] == ["ssh", "-G"] for c in run.calls) and t.proc is None   # nothing but configuration reads
    with pytest.raises(Refused, match="could not read"):
        resolved_host("nope", FakeRun(rc=255))


def test_ssh_commands_for_a_bench_unit(tmp_path):
    run = FakeRun()
    t = SshTransport(Unit("benchpi", "benchpi"), frozenset({"field-pi"}), run=run)
    assert t.check() == "bench-pi.local"
    assert t.argv()[:6] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "benchpi"]
    assert t.argv()[6:] == ["python3", "urchin_bench/bench_agent.py", "urchin_bench", "--runtime", "litert"]
    t.install()
    t.put(tmp_path / "m.tflite", "fit/a/model.tflite")
    copies = [c for c in run.calls if c[0] == "scp"]
    assert copies[0][-1] == "benchpi:urchin_bench/bench_agent.py"
    assert copies[1][-1] == "benchpi:urchin_bench/requirements-litert-2.2.0.txt"
    assert copies[-1][-1] == "benchpi:urchin_bench/models/fit_a_model.tflite"


def test_installing_the_agent_on_a_unit_is_a_logged_rig_change(tmp_path):
    run = FakeRun()
    board = PiBoard(Unit("benchpi", "benchpi"), tmp_path, SshTransport(Unit("benchpi", "benchpi"), frozenset(), run=run))
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


# --- S7 bite 3b-i (QE PR 108 nits, D89's memory floor) ---------------------------------------------------------

from app.bench.board import BAD_ANSWER, LOW_MEMORY, open_board  # noqa: E402

BENCH = {"field_units": ["field-pi"], "units": {"benchpi": {"ssh_host": "benchpi-alias", "cameras": ["n6"]}}}


def test_open_board_gives_a_real_unit_its_pi_board_only_through_bench(tmp_path):
    bench = parse(BENCH)
    board = open_board(bench.unit("benchpi"), "pi", tmp_path, bench=bench, log_dir=tmp_path / "run")
    assert isinstance(board, PiBoard) and board.transport.alias == "benchpi-alias"
    assert board.transport.log == tmp_path / "run" / "agent.log" and board.transport.proc is None   # nothing started
    stranger = Unit("benchpi", "somewhere-else")                    # same name, not what [bench] says
    for unit, kw in ((stranger, {"bench": bench}), (bench.unit("benchpi"), {})):
        with pytest.raises(Refused, match="not given by"):
            open_board(unit, "pi", tmp_path, **kw)
    with pytest.raises(Refused, match="fake settings"):
        open_board(bench.unit("benchpi"), "pi", tmp_path, bench=bench, ms={"m": 1.0})
    with pytest.raises(Refused, match="bites 4-5"):
        open_board(bench.unit("benchpi"), "n6", tmp_path, bench=bench)


def test_an_ssh_link_is_never_made_from_a_bare_alias_or_the_fake_unit():
    for unit in ("benchpi-alias", Bench().unit("fake"), Unit("x")):
        with pytest.raises(Refused, match="listed under"):
            SshTransport(unit, frozenset())
    run = FakeRun(hostname="bench-pi.local")                     # a unit whose own name is a field unit
    with pytest.raises(Refused, match="field unit"):
        SshTransport(Unit("field-pi", "innocent"), frozenset({"field-pi"}), run=run).check()


def test_a_line_that_is_not_json_is_the_bad_answer_stop(tmp_path):
    class Chatty(LocalTransport):
        def argv(self):
            return [sys.executable, "-c", "import sys; sys.stdin.readline(); print('WARNING: delegate'); sys.stdout.flush();"
                    " import time; time.sleep(5)"]
    t = Chatty(tmp_path, timeout_s=5)
    with pytest.raises(BoardError) as e:
        t.start()
    assert e.value.stop == BAD_ANSWER and "WARNING: delegate" in str(e.value)
    t.proc.kill()


def test_the_agents_stderr_goes_to_agent_log_not_a_pipe(tmp_path):
    """200 kB on stderr would fill a pipe nobody reads and block the agent (QE PR 108 nit 4)."""
    class Noisy(LocalTransport):
        def argv(self):
            return [sys.executable, "-c", "import sys, json; sys.stderr.write('w' * 200000); sys.stderr.flush();"
                    " req = json.loads(sys.stdin.readline()); print(json.dumps({'id': req['id'], 'ok': True}));"
                    " sys.stdout.flush()"]
    t = Noisy(tmp_path, timeout_s=10, log=tmp_path / "run" / "agent.log")
    assert t.start() == {}
    assert (tmp_path / "run" / "agent.log").stat().st_size >= 200000
    t.proc.kill()


def test_the_agent_refuses_a_load_and_a_window_under_50_mb_free(tmp_path, monkeypatch):
    folder = tmp_path / "agent"
    agent = ag.Agent(folder, "fake")
    src, sha = model_file(tmp_path)
    (folder / "models").mkdir(parents=True)
    (folder / "models" / "m").write_bytes(src.read_bytes())
    free = {"v": 400 * 1024 * 1024}
    monkeypatch.setattr(ag, "proc_kib", lambda path, key: free["v"] if key == "MemAvailable" else None)
    ok, = talk(agent, {"id": 1, "op": "load", "files": {"m": sha}})
    assert ok["ok"] and ok["loaded"] == ["m"]
    free["v"] = 30 * 1024 * 1024
    window, load = talk(agent, {"id": 2, "op": "time", "name": "m", "n": 3}, {"id": 3, "op": "load", "files": {"m": sha}})
    assert window["stop"] == LOW_MEMORY == load["stop"] and "30.0 MB free" in window["error"]
