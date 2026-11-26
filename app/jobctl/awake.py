"""Keep the laptop awake while a keep-awake job is scheduled, queued or running (D38).

One macOS keep-awake process, `caffeinate -i -s -w <app pid>`: -i prevents idle sleep, -s system
sleep (on AC power), -w ends it when the app's process ends, so a crashed app releases it (man
page on this laptop, read 2026-11-25 by the planning lead, D38). The app starts it when the first
keep-awake job appears and ends it, by the Popen it started and never any other process, when no
such job is left. The app never changes power settings. Lid-closed sleep is not covered: the page
reminds to keep the laptop plugged in with the lid open, and warns when it runs on battery.
"""

from __future__ import annotations

import os
import subprocess
import threading


def default_cmd() -> list[str]:
    return ["caffeinate", "-i", "-s", "-w", str(os.getpid())]


class Awake:
    def __init__(self, cmd: list[str] | None = None):
        self.cmd = cmd or default_cmd()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.error = ""

    def update(self, needed: bool) -> None:
        """Start the keep-awake process if needed and not running; end it if not needed."""
        with self._lock:
            alive = self._proc is not None and self._proc.poll() is None
            if needed and not alive:
                try:
                    self._proc = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL,
                                                  stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.DEVNULL)
                    self.error = ""
                except OSError as e:   # e.g. not macOS: say so on the page, jobs still run
                    self._proc, self.error = None, f"cannot keep the laptop awake: {e}"
            elif not needed and alive:
                self._proc.terminate()   # only the process this app started (D38)
                try:
                    self._proc.wait(5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
                self._proc = None

    def pid(self) -> int | None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc.pid
            return None

    def stop(self) -> None:
        self.update(False)


def on_battery() -> bool | None:
    """True on battery power, False on AC, None if unknown (`pmset -g batt`, read only)."""
    try:
        r = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if "'Battery Power'" in r.stdout:
        return True
    if "'AC Power'" in r.stdout:
        return False
    return None
