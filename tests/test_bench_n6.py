"""S7 bite 4b: the N6 write route (DFU alt 3, read-back, /rom listing, the timing script) and its built-in restore of
ROMFS0 from the backup, against a fake N6 (tests/fake_n6.py: fake mpremote and dfu-util over a 64 KiB partition).
Nothing here touches a board, ssh or the real dfu-util."""

import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from app.bench import riglog, romfs
from app.bench.agent import bench_agent
from app.bench.board import LOAD_FAILED, READBACK, RESTORE_FAILED, BoardError, Model, open_board
from app.bench.n6_board import Backup, N6Board, rom_name
from app.bench.transport import LocalTransport
from app.bench.units import Refused, parse

FAKE = Path(__file__).with_name("fake_n6.py")
P = 64 * 1024
VENDOR = [romfs.Entry("vendor_det.tflite", bytes(range(256)) * 12, 32), romfs.Entry("vendor_det.txt", b"urchin\n", 4)]


def tool(tmp, state, name):
    p = tmp / name
    p.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "{state}" {name} "$@"\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture
def rig(tmp_path):
    """A fake N6 in app mode holding the 'backup' partition, an AE3 port beside it, and the backup file (+4 bytes)."""
    state, ports = tmp_path / "n6", tmp_path / "by-id"
    state.mkdir()
    ports.mkdir()
    (ports / "usb-OpenMV_OpenMV_Camera_AE3-if00").touch()
    (ports / "usb-MicroPython_Pyboard_Virtual_Comm_Port_N6-if00").touch()
    image = romfs.encode(VENDOR)
    partition = image + b"\xff" * (P - len(image))
    (state / "romfs0.bin").write_bytes(partition)
    (state / "backup.bin").write_bytes(partition)
    backup = tmp_path / "n6_ROMFS0.bin"
    backup.write_bytes(partition + b"\x00" * 4)
    data = tmp_path / "data"
    model_file = data / "fit" / "yolox-nano-conv-256-c5" / "n6" / "model_int8.tflite"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"NBIN" + bytes(3000))
    extra = ("--mpremote", tool(tmp_path, state, "mpremote"), "--dfu-util", tool(tmp_path, state, "dfu-util"),
             "--ports", str(ports / "*"), "--partition", str(P), "--back-s", "2")
    unit = parse({}).unit("fake")
    board = N6Board(unit, data, LocalTransport(tmp_path / "agent", extra=extra, timeout_s=30), work_dir=tmp_path / "run",
                    backup=backup, backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(), partition=P)
    return {"board": board, "state": state, "data": data, "partition": partition,
            "model": Model("yolox-nano-conv-256-c5", model_file), "tmp": tmp_path}


def calls(state):
    return [json.loads(line) for line in (state / "calls.log").read_text().splitlines()] if (state / "calls.log").exists() else []


def fault(state, **f):
    (state / "faults.json").write_text(json.dumps(f))


def only_alt3_written(state):
    writes = [c for c in calls(state) if c[0] == "dfu-util" and "-D" in c]
    return all(c[c.index("-a") + 1] == "3" for c in writes) and not list(state.glob("alt*_written"))


def test_a_load_writes_the_backups_entries_plus_ours_and_times_it(rig):
    b, state = rig["board"], rig["state"]
    done = b.load(rig["model"])
    try:
        entries, length = romfs.decode((state / "romfs0.bin").read_bytes())
        assert [e.name for e in entries] == ["vendor_det.tflite", "vendor_det.txt", "yolox-nano-conv-256-c5.tflite"]
        assert entries[:2] == VENDOR and entries[2].data == rig["model"].path.read_bytes()      # D82a, byte for byte
        assert done["rom"]["yolox-nano-conv-256-c5.tflite"] == 3004 and done["route"] == "ROMFS0, DFU alt 3"
        assert (state / "mode").read_text() == "app"                                           # booted again
        t = b.time("yolox-nano-conv-256-c5", 300)
        assert t["n"] == 300 and t["warmups"] == 2 and t["ms_mean"] == pytest.approx(10.6) and t["ms_max"] == 10.7
        assert t["mem_before_load"] == 25_000_000 and t["mem_after_load"] == 24_000_000
        info = b.info()
        assert info["kind"] == "n6" and info["board_type"] == "N6" and "yolox-nano-conv-256-c5.tflite" in info["rom"]
    finally:
        b.close()
    assert only_alt3_written(state)
    assert not any(c[0] == "mpremote" and "OpenMV_Camera" in c[2] for c in calls(state))   # the AE3 was not asked
    assert [r["state"] for r in riglog.read(rig["data"]) if r["what"].startswith("load")] == ["done"]


@pytest.mark.parametrize("faults, stop", [({"corrupt_write": 1}, READBACK), ({"write_fails": 1}, LOAD_FAILED),
                                          ({"hide": "yolox-nano-conv-256-c5.tflite"}, LOAD_FAILED)])
def test_a_failed_load_puts_the_backup_back_by_itself(rig, faults, stop):
    b, state = rig["board"], rig["state"]
    fault(state, **faults)                       # "hide": /rom lacks ours, so the listing check fails; the backup's passes
    with pytest.raises(BoardError) as e:
        b.load(rig["model"])
    b.close()
    assert e.value.stop == stop
    assert "put back from the backup: read back identical" in str(e.value)
    assert (state / "romfs0.bin").read_bytes() == rig["partition"] and (state / "mode").read_text() == "app"
    assert only_alt3_written(state)
    rows = riglog.read(rig["data"])
    assert [r["state"] for r in rows if r["what"].startswith("restore ROMFS0")] == ["done"]
    assert [r["state"] for r in rows if r["what"].startswith("load")] == ["failed"]


def test_a_fault_in_a_window_restores_too(rig):
    b, state = rig["board"], rig["state"]
    b.load(rig["model"])
    fault(state, time_fails=True)
    with pytest.raises(BoardError) as e:
        b.time("yolox-nano-conv-256-c5", 10)
    b.close()
    assert e.value.stop == LOAD_FAILED and "Failed to load network" in str(e.value) and "put back" in str(e.value)
    assert (state / "romfs0.bin").read_bytes() == rig["partition"] and len(b.restores) == 1


def test_a_board_that_does_not_come_back_is_restore_failed(rig):
    """Our image boots into nothing (no serial, no DFU): the restore cannot run, and the job says so loudly."""
    b, state = rig["board"], rig["state"]
    fault(state, dead_unless_backup=True)
    with pytest.raises(BoardError) as e:
        b.load(rig["model"])
    b.close()
    assert e.value.stop == RESTORE_FAILED and "did not verify" in str(e.value)
    assert not any(c[0] == "mpremote" and "OpenMV_Camera" in c[2] for c in calls(state))   # 60 s of waiting: AE3 untouched
    assert [r["state"] for r in riglog.read(rig["data"]) if r["what"].startswith("restore ROMFS0")] == ["failed"]


def test_nothing_reaches_the_rig_when_the_backup_is_not_the_recorded_one(rig):
    b, state = rig["board"], rig["state"]
    b.backup_sha256 = "0" * 64
    with pytest.raises(BoardError, match="not the recorded"):
        b.load(rig["model"])
    b.close()
    assert calls(state) == [] and not b.restores


def test_a_backup_that_does_not_re_encode_is_refused(tmp_path):
    """QE PR 119 nit 1: encode(decode(x)) == x[:length] on the backup before anything is built."""
    odd = romfs.encode([romfs.Entry("label.txt", b"x" * 5, 32)])          # aligned as decode would not guess
    raw = odd + b"\xff" * (P - len(odd)) + b"\x00" * 4
    f = tmp_path / "b.bin"
    f.write_bytes(raw)
    with pytest.raises(BoardError, match="known-good"):
        Backup(f, hashlib.sha256(raw).hexdigest(), P)


def test_the_agent_writes_no_alt_but_3(tmp_path):
    n6 = bench_agent.N6(str(tmp_path), "mpremote", "dfu-util", "", lambda d: None)
    for alt in (0, 1, 2):
        with pytest.raises(bench_agent.Stop, match="only alt 3"):
            n6.write(str(tmp_path / "x"), alt=alt)


def test_rom_names_and_the_real_unit_wiring(tmp_path):
    assert rom_name("yolox-nano-conv-256-c5") == "yolox-nano-conv-256-c5.tflite" and rom_name("a/b.tflite") == "a_b.tflite"
    bench = parse({"field_units": ["field-pi"], "units": {"benchpi": {"ssh_host": "benchpi", "cameras": ["n6"]}}})
    board = open_board(bench.unit("benchpi"), "n6", tmp_path, bench=bench, log_dir=tmp_path / "run")
    assert isinstance(board, N6Board) and board.transport.unit.name == "benchpi"       # built; nothing connected
    with pytest.raises(Refused, match="not built yet"):
        open_board(bench.unit("benchpi"), "ae3", tmp_path, bench=bench)
    pi_only = parse({"field_units": ["field-pi"], "units": {"benchpi": {"ssh_host": "benchpi", "cameras": []}}})
    with pytest.raises(Refused, match="lists no n6"):
        open_board(pi_only.unit("benchpi"), "n6", tmp_path, bench=pi_only)


class _Script(LocalTransport):
    def __init__(self, folder, code):
        super().__init__(folder, timeout_s=2)
        self.code = code

    def argv(self):
        return [sys.executable, "-c", self.code]


HELLO = "import sys, json\nfor line in sys.stdin:\n    r = json.loads(line)\n"


def test_a_late_line_from_an_earlier_request_is_dropped(tmp_path):
    """QE PR 119 nit 2: an answer to request 1 arriving during request 2 is not taken for request 2's answer."""
    code = HELLO + ('    if r["id"] > 1: print(json.dumps({"id": r["id"] - 1, "alive": True}))\n'
                    '    print(json.dumps({"id": r["id"], "ok": True, "op": r["op"]}), flush=True)\n')
    t = _Script(tmp_path, code)
    t.start()
    assert t.request("info")["op"] == "info"
    t.close()


def test_an_agent_that_ends_says_so(tmp_path):
    """QE PR 119 nit 3: stdout closed is 'the agent ended', with the log's tail, not 'within 60 s'."""
    code = HELLO + ('    if r["op"] == "hello": print(json.dumps({"id": r["id"], "ok": True}), flush=True)\n'
                    '    else: sys.stderr.write("Traceback: boom\\n"); sys.exit(1)\n')
    t = _Script(tmp_path, code)
    t.start()
    t0 = time.monotonic()
    with pytest.raises(BoardError, match="ended before it answered 'info'.*boom"):
        t.request("info")
    assert time.monotonic() - t0 < 1.5
    t.close()
