"""S7 bite 3b-ii (D89): the agent builds the runtime's environment from a hash-pinned file, inside its folder only;
the Pi board installs it once, as a logged rig change, then runs the agent from it. Fake install and fake transport only:
no venv, no pip, no network, no unit."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from app.bench import riglog
from app.bench.agent import bench_agent as ag
from app.bench.board import Model
from app.bench.pi_board import PiBoard
from app.bench.transport import AGENT_FILES, LocalTransport, SshTransport
from app.bench.units import Bench, Unit

REQ = Path(ag.__file__).parent / "requirements-litert-2.2.0.txt"


def agent_in(tmp_path):
    folder = tmp_path / "agent"
    (folder / "models").mkdir(parents=True)
    (folder / REQ.name).write_bytes(REQ.read_bytes())
    return ag.Agent(folder, "fake"), folder


def test_the_pin_file_pins_every_package_by_version_and_hash():
    lines = [l for l in REQ.read_text().splitlines() if l and not l.startswith("#")]
    pins = dict(re.fullmatch(r"([a-z0-9-]+)==([0-9.]+) --hash=sha256:[0-9a-f]{64}", l).groups() for l in lines)
    assert pins == {"ai-edge-litert": "2.2.0", "backports-strenum": "1.2.8", "flatbuffers": "25.12.19",
                    "ml-dtypes": "0.7.0", "numpy": "2.5.5", "protobuf": "7.36.2", "tqdm": "4.70.1",
                    "typing-extensions": "4.16.0"}


def test_install_uses_pypi_only_hashes_and_keeps_every_write_in_the_folder(tmp_path, monkeypatch):
    agent, folder = agent_in(tmp_path)
    seen = []

    def fake_run(argv, env, alive, cwd):
        seen.append((argv, env))
        if argv[1:3] == ["-m", "venv"]:
            os.makedirs(Path(argv[3]) / "bin")
        return 0, json.dumps({"numpy": ["2.5.5", "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"]}) if "-c" in argv else ""
    monkeypatch.setattr(ag, "run_steps", fake_run)
    assert agent.op_install_status({"runtime": "litert"})["ready"] is False
    done = agent.op_install({"runtime": "litert"})["installed"]
    venv, pip, listing = seen
    assert venv[0][1:] == ["-m", "venv", str(folder / "venv-litert-2.2.0")]
    for flag in ("--no-cache-dir", "--only-binary=:all:", "--require-hashes"):
        assert flag in pip[0]
    assert pip[0][pip[0].index("--index-url") + 1] == "https://pypi.org/simple" and pip[0][-1] == str(folder / REQ.name)
    env = pip[1]
    assert env["PIP_CONFIG_FILE"] == os.devnull and env["HOME"] == str(folder) and env["TMPDIR"].startswith(str(folder))
    assert done["packages"]["numpy"][0] == "2.5.5" and done["venv"] == "venv-litert-2.2.0"
    status = agent.op_install_status({"runtime": "litert"})
    assert status["ready"] and status["installed"]["requirements_sha256"] == hashlib.sha256(REQ.read_bytes()).hexdigest()
    (folder / REQ.name).write_text(REQ.read_text() + "# changed\n")      # a new pin file: not ready any more
    assert agent.op_install_status({"runtime": "litert"})["ready"] is False


def test_no_venv_on_the_pi_stops_the_install_and_leaves_nothing_half_made(tmp_path, monkeypatch):
    agent, folder = agent_in(tmp_path)

    def no_venv(argv, env, alive, cwd):
        os.makedirs(Path(argv[3]) / "bin")
        return 1, "The virtual environment was not created successfully because ensurepip is not available."
    monkeypatch.setattr(ag, "run_steps", no_venv)
    with pytest.raises(ag.Stop) as e:
        agent.op_install({"runtime": "litert"})
    assert e.value.stop == ag.INSTALL_FAILED and "no apt" in str(e.value) and "ensurepip" in str(e.value)
    assert not (folder / "venv-litert-2.2.0").exists()
    (folder / REQ.name).unlink()
    with pytest.raises(ag.Stop, match="copy the agent again"):
        agent.op_install({"runtime": "litert"})


def test_install_sends_alive_lines_while_pip_is_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "ALIVE_S", 0.2)
    beats = []
    code, _ = ag.run_steps([os.sys.executable, "-c", "import time; time.sleep(1)"], dict(os.environ), beats.append, tmp_path)
    assert code == 0 and len(beats) >= 3


def test_the_pi_board_installs_once_then_runs_from_the_environment(tmp_path):
    data = tmp_path / "data"
    unit = Bench().unit("fake")
    src = tmp_path / "m.tflite"
    src.write_bytes(b"model")

    def board():
        return PiBoard(unit, data, LocalTransport(tmp_path / "agent", extra=("--fake-install",)), install="litert")
    b = board()
    try:
        b.load(Model("m", src))
        assert b.time("m", 3)["n"] == 3
        assert b.hello["runtime_install"]["venv"] == "venv-litert-2.2.0"
        assert b.transport.python == str(tmp_path / "agent" / "venv-litert-2.2.0" / "bin" / "python")
    finally:
        b.close()
    whats = [r["what"] for r in reversed(riglog.read(data))]
    assert whats == ["copy the bench agent", "install the litert runtime into ~/urchin_bench/venv-litert-2.2.0", "load m"]
    b = board()                                                   # same folder again: nothing to copy or install
    try:
        b.info()
    finally:
        b.close()
    assert len(riglog.read(data)) == 3


def test_ssh_agent_check_reads_sha256sum_and_runs_from_the_venv():
    class Run:
        def __init__(self, sums):
            self.sums, self.calls = sums, []

        def __call__(self, argv, **kw):
            self.calls.append(argv)
            if argv[:2] == ["ssh", "-G"]:
                return subprocess.CompletedProcess(argv, 0, "hostname bench-pi.local\n", "")
            if "sha256sum" in argv:
                return subprocess.CompletedProcess(argv, 0, self.sums, "")
            return subprocess.CompletedProcess(argv, 0, "", "")
    good = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  urchin_bench/{f.name}\n" for f in AGENT_FILES)
    t = SshTransport(Unit("benchpi", "benchpi"), frozenset(), run=Run(good))
    assert t.agent_current() is True
    assert SshTransport(Unit("benchpi", "benchpi"), frozenset(), run=Run(good.replace("a", "b", 1))).agent_current() is False
    v = t.with_python(t.venv_python("venv-litert-2.2.0"))
    assert v.argv()[6] == "urchin_bench/venv-litert-2.2.0/bin/python" and t.argv()[6] == "python3"
    t.install()
    assert [c[-1] for c in t.run.calls if c[0] == "scp"] == [f"benchpi:urchin_bench/{f.name}" for f in AGENT_FILES]


def test_a_pi_card_states_its_one_thread(tmp_path):
    """D88: the Pi's first LiteRT card runs 1 thread, stated on the card; a camera card has no thread count."""
    from app.bench import card, units
    from tests.test_bench_cards import form
    d = tmp_path / "data"
    (d / "fit" / "yolox_nano" / "int8").mkdir(parents=True)
    (d / "fit" / "yolox_nano" / "int8" / "model_int8.tflite").write_bytes(b"model")
    assert card.build(d, units.Bench(), form(board="pi"))["threads"] == 1
    assert "threads" not in card.build(d, units.Bench(), form(board="n6"))


def test_a_transport_with_no_log_path_writes_outside_the_checkout(tmp_path, monkeypatch):
    """QE PR 112 nit 3: the fallback agent.log is a temporary folder, never the current folder."""
    monkeypatch.chdir(tmp_path)
    t = LocalTransport(tmp_path / "agent")
    t.log = None
    t.start()
    try:
        assert t.log.name == "agent.log" and tmp_path not in t.log.parents and not (tmp_path / "agent.log").exists()
    finally:
        t.close()
