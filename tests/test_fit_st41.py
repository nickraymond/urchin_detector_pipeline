"""S6 bite 2f (D71): the ST Edge AI Core 4.1.0 step may only run with "n" answered, the network denied and its
statistics file unwritable. A shell script stands in for ST's tool; nothing of ST's runs here."""

import shutil
import socket
import sys
from pathlib import Path

import pytest

from app.fit import convert, n6_st41
from app.fit.convert_worker import ToolsMissing

REPORT = Path(__file__).parent / "fixtures" / "fit" / "stedgeai_yolox_nano_epochs.txt"
needs_sandbox = pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("sandbox-exec"),
                                   reason="macOS sandbox-exec only")


def test_the_profile_denies_the_network_and_the_statistics_file(tmp_path):
    stats = tmp_path / ".stmai"
    cmd = n6_st41.sandboxed(["/x/stedgeai", "generate"], stats)
    assert cmd[:2] == ["sandbox-exec", "-p"] and cmd[3:] == ["/x/stedgeai", "generate"]
    assert "(deny network*)" in cmd[2] and f'(deny file-write* (literal "{stats}"))' in cmd[2]
    assert convert.COMPILERS["n6_st41"] == "app.fit.n6_st41" and "n6_st41" not in convert.CAMERAS   # never a column
    assert convert.fit_settings(None)["stedgeai41"].endswith("/4.1/Utilities/macarm/stedgeai")


def test_the_step_refuses_to_start_the_tool_without_the_block(tmp_path):
    started = tmp_path / "started"
    tool = tmp_path / "stedgeai"
    tool.write_text(f"#!/bin/sh\ntouch {started}\n")
    tool.chmod(0o755)
    stats = tmp_path / ".stmai"
    for cmd in ([str(tool), "generate"],                                                    # not wrapped at all
                ["sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", str(tool)],     # file not blocked
                ["sandbox-exec", "-p", f'(version 1)(allow default)(deny file-write* (literal "{stats}"))', str(tool)]):
        with pytest.raises(RuntimeError, match="Nothing was started"):
            n6_st41.run(cmd, tmp_path, stats)
    assert not started.exists()


@needs_sandbox
def test_a_run_answers_n_cannot_reach_the_network_and_cannot_write_the_statistics_file(tmp_path):
    stats = tmp_path / ".stmai"
    stats.write_text("[statistics]\nuuid = x\nstacked_stats = []\n")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    tool = tmp_path / "stedgeai"
    tool.write_text("#!/bin/sh\nread answer\necho \"answer=$answer\"\n"
                    f"echo 'stacked_stats = [1]' >> {stats} 2>/dev/null && echo wrote || echo write-refused\n"
                    f"{sys.executable} -c \"import socket; socket.create_connection(('127.0.0.1', {port}), 2)\" "
                    "2>/dev/null && echo connected || echo network-refused\n")
    tool.chmod(0o755)
    done, note = n6_st41.run(n6_st41.sandboxed([str(tool)], stats), tmp_path, stats)
    server.close()
    assert "answer=n" in done.stdout and "write-refused" in done.stdout and "network-refused" in done.stdout
    assert stats.read_text().endswith("stacked_stats = []\n") and "unchanged" in note and str(stats) in note


@needs_sandbox
def test_compile_reports_placement_only_and_logs_the_checksum(tmp_path, monkeypatch):
    stats = tmp_path / ".stmai"
    stats.write_text("[statistics]\n")
    monkeypatch.setattr(n6_st41, "STATS_FILE", stats)
    monkeypatch.setattr(n6_st41.run, "__defaults__", (stats,))
    monkeypatch.setattr(n6_st41.sandboxed, "__defaults__", (stats,))
    omv = tmp_path / "omv" / "lib" / "stai" / "scripts"
    omv.mkdir(parents=True)
    (omv / "neuralart.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    tool = tmp_path / "stedgeai"
    tool.write_text("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'ST Edge AI Core v4.1.0-20732 abc'; exit 0; fi\n"
                    f"mkdir -p {out}/gen && cp {REPORT} {out}/gen/network_generate_report.txt\n")
    tool.chmod(0o755)
    job = {"fit": {**convert.fit_settings({"openmv_dir": str(tmp_path / "omv")}), "stedgeai41": str(tool)}}
    model = tmp_path / "model_int8.tflite"
    model.write_bytes(b"x")
    fit = n6_st41.compile(model, out, job)
    assert fit["converts"] and fit["tool"].startswith("ST Edge AI Core v4.1.0") and fit["file_bytes"] is None
    assert fit["placement"]["unit"] == "epochs" and fit["placement"]["cpu"] == 0 and "placement only" in fit["note"]
    assert "unchanged" in fit["stats_file"] and "sha256 before" in (out / "compile.log").read_text()
    tool.write_text("#!/bin/sh\necho 'NOT IMPLEMENTED: Layer of type LESS is currently not supported'; exit 255\n")
    failed = n6_st41.compile(model, out, job)
    assert failed["converts"] is False and "LESS" in failed["error"]
    with pytest.raises(ToolsMissing, match="installed by Nick"):
        n6_st41.compile(model, out, {"fit": {**job["fit"], "stedgeai41": str(tmp_path / "none")}})
