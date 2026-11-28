"""How the laptop reaches the bench agent (S7 bite 3, D81): a child process that speaks JSON lines.

    LocalTransport   the agent as a child process on the laptop, in a scratch folder: the "fake transport" every
                     test, builder and QE uses. Nothing leaves the laptop.
    SshTransport     `ssh <alias> python3 <folder>/bench_agent.py <folder>` and `scp` for files, for a bench unit
                     listed under [bench]. Builders and QE never open one (SPEC constraint 9); only the rig test
                     session does, through the app, after QE has reviewed its card.

Before an SshTransport starts, the host that `ssh -G <alias>` resolves the alias to is checked against field_units
too (QE PR 98 nit 5): an alias that names a bench unit but points at a field unit is refused. `ssh -G` only prints
ssh's configuration; it opens no connection.

A request that gets no answer in `timeout_s` (60 s, D84) is the "no_answer" stop rule.
"""

from __future__ import annotations

import itertools
import json
import selectors
import shutil
import subprocess
import sys
from pathlib import Path

from app.bench.agent.bench_agent import file_name
from app.bench.board import NO_ANSWER, BoardError
from app.bench.units import Refused, _is_field

AGENT = Path(__file__).resolve().parent / "agent" / "bench_agent.py"
REMOTE_FOLDER = "urchin_bench"            # the agent's folder in the bench Pi's home directory
TIMEOUT_S = 60.0


class Transport:
    """One agent process. request() sends one op and returns its answer; a refusal comes back as BoardError."""

    proc: subprocess.Popen | None = None
    timeout_s = TIMEOUT_S

    def argv(self) -> list[str]:
        raise NotImplementedError

    def put(self, local: Path, name: str) -> None:
        """Copy a file into the agent's models/ folder."""
        raise NotImplementedError

    def start(self) -> dict:
        self.proc = subprocess.Popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self._ids = itertools.count(1)
        return self.request("hello")

    def request(self, op: str, **body) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            raise BoardError(NO_ANSWER, f"the bench agent is not running ({self.where()}); start it again")
        rid = next(self._ids)
        self.proc.stdin.write(json.dumps({"id": rid, "op": op, **body}) + "\n")
        self.proc.stdin.flush()
        line = self._readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.poll() is not None else ""
            raise BoardError(NO_ANSWER, f"no answer from the bench agent to {op!r} within {self.timeout_s:g} s "
                                        f"({self.where()}){': ' + err.strip()[-300:] if err.strip() else ''}")
        answer = json.loads(line)
        if answer.get("id") != rid:
            raise BoardError(NO_ANSWER, f"the bench agent answered request {answer.get('id')}, not {rid}")
        if not answer.get("ok"):
            raise BoardError(answer.get("stop") or NO_ANSWER, answer.get("error") or "the agent refused")
        return {k: v for k, v in answer.items() if k not in ("id", "ok")}

    def _readline(self) -> str:
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        ready = sel.select(self.timeout_s)
        sel.close()
        return self.proc.stdout.readline() if ready else ""

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.request("bye")
            except BoardError:
                pass
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def where(self) -> str:
        raise NotImplementedError


class LocalTransport(Transport):
    """The agent on the laptop, in `folder`, with `python` (any Python 3; the app's own by default)."""

    def __init__(self, folder: Path, runtime: str = "fake", python: str = sys.executable, timeout_s: float = TIMEOUT_S):
        self.folder, self.runtime, self.python, self.timeout_s = Path(folder), runtime, python, timeout_s

    def argv(self) -> list[str]:
        return [self.python, str(AGENT), str(self.folder), "--runtime", self.runtime]

    def put(self, local: Path, name: str) -> None:
        (self.folder / "models").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, self.folder / "models" / file_name(name))

    def where(self) -> str:
        return f"local agent in {self.folder}"


def resolved_host(alias: str, run=subprocess.run) -> str:
    """The hostname ssh would connect to for `alias` (`ssh -G` prints the configuration; no connection)."""
    done = run(["ssh", "-G", alias], capture_output=True, text=True, timeout=10)
    if done.returncode != 0:
        raise Refused(f"ssh could not read its configuration for {alias!r}: {done.stderr.strip()[-200:]}. "
                      "Fix: check the alias in ~/.ssh/config.")
    for line in done.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key.lower() == "hostname" and value.strip():
            return value.strip()
    raise Refused(f"ssh -G {alias} names no hostname; the unit is not used.")


class SshTransport(Transport):
    """The agent on a bench Pi. `check()` must pass before start(); start() calls it."""

    def __init__(self, alias: str, field_units: frozenset[str], runtime: str = "litert", run=subprocess.run,
                 timeout_s: float = TIMEOUT_S):
        self.alias, self.field_units, self.runtime, self.run, self.timeout_s = (
            alias, field_units, runtime, run, timeout_s)

    def check(self) -> str:
        host = resolved_host(self.alias, self.run)
        if _is_field(host, self.field_units):
            raise Refused(f"the ssh alias {self.alias!r} resolves to {host!r}, a field unit: the app never touches "
                          "field units (SPEC constraint 9). Fix: point the alias at the bench Pi.")
        return host

    def ssh(self, *command: str) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.alias, *command]

    def argv(self) -> list[str]:
        return self.ssh("python3", f"{REMOTE_FOLDER}/bench_agent.py", REMOTE_FOLDER, "--runtime", self.runtime)

    def start(self) -> dict:
        self.check()
        return super().start()

    def install(self) -> None:
        """Copy the agent into its folder on the Pi (a rig change: PiBoard logs it)."""
        self.check()
        self._run(self.ssh("mkdir", "-p", f"{REMOTE_FOLDER}/models"), "make the agent's folder")
        self._run(["scp", "-q", "-o", "BatchMode=yes", str(AGENT), f"{self.alias}:{REMOTE_FOLDER}/bench_agent.py"],
                  "copy the agent")

    def put(self, local: Path, name: str) -> None:
        self.check()
        self._run(["scp", "-q", "-o", "BatchMode=yes", str(local), f"{self.alias}:{REMOTE_FOLDER}/models/{file_name(name)}"],
                  f"copy {file_name(name)}")

    def _run(self, argv: list[str], what: str) -> None:
        done = self.run(argv, capture_output=True, text=True, timeout=self.timeout_s)
        if done.returncode != 0:
            raise BoardError(NO_ANSWER, f"could not {what} on {self.alias}: {done.stderr.strip()[-300:]}")

    def where(self) -> str:
        return f"{self.alias}:~/{REMOTE_FOLDER}"
