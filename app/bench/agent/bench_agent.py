"""The bench agent (S7 bite 3, D81): runs ON the bench Pi, in its own folder, started by the laptop over ssh as
`python3 bench_agent.py <folder> [--runtime litert|fake]`. Plain Python 3 and its standard library; no import from
the app (it is copied alone). numpy and the runtime are imported only when a model is loaded.

Protocol: JSON lines. The laptop writes one request per line, {"id": n, "op": ..., ...}; the agent answers each
with one line, {"id": n, "ok": true, ...} or {"id": n, "ok": false, "stop": <stop rule>, "error": <words>}.
Ops: hello, info, load, time, reset, bye, install_status, install (D89). Model files are copied into <folder>/models/ by the laptop before
"load"; the agent only checks their sha256 (a mismatch is the "readback_differs" stop rule, D84).

Timing (D83): one seeded synthetic input at the model's input size; 2 warm-ups; time.perf_counter_ns() around
each invoke of the window; peak memory is the process's VmHWM. Threads are fixed per load and recorded. While a
window runs the agent writes {"id": n, "alive": true, "done": k} after the first inference that ends ALIVE_S
seconds or more after the last such line (D84: after each inference at the latest once 10 s have passed), so a slow model (300
inferences at a second each) is not taken for a board that stopped answering. `time` returns the fields every
Board.time returns (n, warmups, ms_mean, ms_min, ms_max, mem_before_load, mem_after_load, mem_after: free memory,
MemAvailable) plus vm_hwm.
The cameras (n6, ae3) are bites 4 and 5: asked for one, the agent says so.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time

VERSION = "1"
WARMUPS = 2
ALIVE_S = 5.0          # an "alive" line this often during a window: D84 asks at least every 10 s; 60 s is for silence
SEED = 0
NO_ANSWER, READBACK, LOAD_FAILED, NOT_LOADED, NOT_BUILT, LOW_MEMORY = (
    "no_answer", "readback_differs", "load_failed", "not_loaded", "not_built", "low_memory")
INSTALL_FAILED = "install_failed"
MEM_FLOOR = 50 * 1024 * 1024   # D89: no load and no window when MemAvailable is under 50 MB


def file_name(name):
    """The file a model is copied to in <folder>/models/: its name with anything but letters, digits, '.', '_' and
    '-' replaced by '_' (the laptop uses the same rule; a name such as "fit/x/model.tflite" stays unique)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class Stop(Exception):
    def __init__(self, stop, message):
        super().__init__(message)
        self.stop = stop


def proc_kib(path, key):
    """A "<key>: <n> kB" line of a /proc file, in bytes; None where there is no /proc (the laptop)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FakeRuntime:
    """No runtime and no numpy: invoke does nothing (or sleeps `ms`). For the laptop's tests of the protocol."""
    name = "fake"
    ms = 0.0

    def open(self, path, threads):
        return {"path": path}

    def invoke(self, model):
        if self.ms:
            time.sleep(self.ms / 1000)

    def describe(self, model):
        return {"input": None, "outputs": []}


class LiteRT:
    """LiteRT (ai-edge-litert, Apache-2.0; D85) on the 8-bit TFLite file S6 makes."""
    name = "litert"

    def open(self, path, threads):
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path, num_threads=threads)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        rng = np.random.default_rng(SEED)
        dtype = np.dtype(inp["dtype"])
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            data = rng.integers(info.min, info.max, size=inp["shape"], endpoint=True, dtype=dtype)
        else:
            data = rng.uniform(0.0, 1.0, size=inp["shape"]).astype(dtype)
        it.set_tensor(inp["index"], data)
        return {"it": it, "inp": inp, "np": np}

    def invoke(self, model):
        model["it"].invoke()

    def describe(self, model):
        np = model["np"]
        shape = lambda d: {"shape": [int(v) for v in d["shape"]], "dtype": np.dtype(d["dtype"]).name}
        return {"input": shape(model["inp"]), "outputs": [shape(o) for o in model["it"].get_output_details()]}

    def version(self):
        from importlib import metadata
        return "ai-edge-litert " + metadata.version("ai-edge-litert")


RUNTIMES = {"fake": FakeRuntime, "litert": LiteRT}

# D89: a runtime's own environment inside the agent's folder, from a hash-pinned file the laptop copies next to the agent
INSTALLS = {"litert": {"venv": "venv-litert-2.2.0", "requirements": "requirements-litert-2.2.0.txt"}}
PIP_ARGS = ["install", "--no-cache-dir", "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
            "--require-hashes", "--no-input", "-r"]
FAKE_INSTALL = False            # tests only (--fake-install): make the environment without venv or pip


def run_steps(argv, env, alive, cwd):
    """Run one install command, writing an alive line every ALIVE_S seconds even while it prints nothing (pip is
    silent while it downloads); returns (exit code, the output's tail)."""
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd) as p:
        out = []
        reader = threading.Thread(target=lambda: out.extend(p.stdout), daemon=True)
        reader.start()
        last = time.monotonic()
        while p.poll() is None:
            time.sleep(0.2)
            if time.monotonic() - last >= ALIVE_S:
                alive(len(out))
                last = time.monotonic()
        reader.join(5)
        return p.returncode, "".join(out)[-1500:]


def check_memory(what):
    """The D89 floor: refuse `what` when free memory is known and under 50 MB (unknown on the laptop: no check)."""
    free = proc_kib("/proc/meminfo", "MemAvailable")
    if free is not None and free < MEM_FLOOR:
        raise Stop(LOW_MEMORY, f"only {free / 1048576:.1f} MB free on the Pi (MemAvailable); {what} needs at least "
                               f"{MEM_FLOOR // 1048576} MB free (D89)")
    return free


class Agent:
    def __init__(self, folder, runtime="litert"):
        self.folder = os.path.abspath(folder)
        self.models_dir = os.path.join(self.folder, "models")
        self.runtime = RUNTIMES[runtime]()
        self.loaded = {}          # name -> (runtime model, threads)
        self.mem_before_load = self.mem_after_load = None
        self.alive = lambda done: None

    def handle(self, req):
        op = req.get("op")
        fn = getattr(self, "op_" + str(op), None)
        if fn is None:
            raise Stop(LOAD_FAILED, f"unknown op {op!r}")
        kind = req.get("kind", "pi")
        if op in ("info", "load", "time", "reset") and kind != "pi":
            raise Stop(NOT_BUILT, f"the {kind} route is not built yet (S7 bites 4-5); this agent runs the Pi only")
        return fn(req)

    def op_hello(self, req):
        return {"agent": VERSION, "python": platform.python_version(), "machine": platform.machine(),
                "system": platform.platform(), "folder": self.folder, "runtime": self.runtime.name}

    def op_info(self, req):
        version = getattr(self.runtime, "version", lambda: self.runtime.name)()
        return {"kind": "pi", "firmware": platform.platform(), "runtime": version,
                "mem_free": proc_kib("/proc/meminfo", "MemAvailable"), "model_area_bytes": 0}

    def op_load(self, req):
        threads = int(req.get("threads", 1))
        self.loaded = {}
        self.mem_before_load = check_memory("a load")
        for name, want in sorted(req["files"].items()):
            path = os.path.join(self.models_dir, file_name(name))
            if not os.path.isfile(path):
                raise Stop(LOAD_FAILED, f"{name} is not in {self.models_dir}: the copy did not arrive")
            have = sha256(path)
            if have != want:
                raise Stop(READBACK, f"{name} on the Pi has sha256 {have[:16]}…, the laptop sent {want[:16]}…")
            try:
                self.loaded[name] = (self.runtime.open(path, threads), threads)
            except Exception as e:                     # the runtime's own words go back to the laptop
                raise Stop(LOAD_FAILED, f"{self.runtime.name} could not load {name}: {type(e).__name__}: {e}")
        self.mem_after_load = proc_kib("/proc/meminfo", "MemAvailable")
        return {"loaded": sorted(self.loaded), "route": "agent folder", "threads": threads,
                "io": {n: self.runtime.describe(m) for n, (m, _) in self.loaded.items()},
                "mem_free": proc_kib("/proc/meminfo", "MemAvailable"), "vm_hwm": proc_kib("/proc/self/status", "VmHWM")}

    def op_time(self, req):
        name, n = req["name"], int(req["n"])
        if name not in self.loaded:
            raise Stop(NOT_LOADED, f"{name} is not loaded on the Pi")
        if n < 1:
            raise Stop(LOAD_FAILED, "a window needs at least one inference")
        model, threads = self.loaded[name]
        check_memory("a window")
        for _ in range(WARMUPS):
            self.runtime.invoke(model)
        each, last = [], time.monotonic()
        for _ in range(n):
            t = time.perf_counter_ns()
            self.runtime.invoke(model)
            each.append((time.perf_counter_ns() - t) / 1e6)
            if time.monotonic() - last >= ALIVE_S:
                self.alive(len(each))
                last = time.monotonic()
        return {"n": n, "warmups": WARMUPS, "threads": threads, "ms_mean": sum(each) / n, "ms_min": min(each),
                "ms_max": max(each), "mem_before_load": self.mem_before_load, "mem_after_load": self.mem_after_load,
                "mem_after": proc_kib("/proc/meminfo", "MemAvailable"), "vm_hwm": proc_kib("/proc/self/status", "VmHWM")}

    def op_reset(self, req):
        self.loaded = {}
        return {"reset": True}

    def _install(self, req):
        name = req.get("runtime", "litert")
        if name not in INSTALLS:
            raise Stop(INSTALL_FAILED, f"no install is defined for runtime {name!r}")
        spec = INSTALLS[name]
        venv = os.path.join(self.folder, spec["venv"])
        req_file = os.path.join(self.folder, spec["requirements"])
        if not os.path.isfile(req_file):
            raise Stop(INSTALL_FAILED, f"{spec['requirements']} is not in {self.folder}: copy the agent again")
        return spec, venv, req_file, os.path.join(venv, "installed.json")

    def op_install_status(self, req):
        spec, venv, req_file, record = self._install(req)
        try:
            with open(record) as f:
                done = json.load(f)
        except (OSError, ValueError):
            return {"ready": False, "venv": spec["venv"]}
        return {"ready": done.get("requirements_sha256") == sha256(req_file), "venv": spec["venv"], "installed": done}

    def op_install(self, req):
        """Build the runtime's environment (D89). Every write stays inside the agent's folder: the venv, pip's
        temporary files (TMPDIR) and no cache; pip reads no config file (the Pi's points at piwheels too)."""
        spec, venv, req_file, record = self._install(req)
        check_memory("an install")                    # pip on a 512 MB Pi (QE PR 113 nit 4)
        if os.path.isdir(venv):
            shutil.rmtree(venv)                       # a half-made environment from an earlier try
        tmp = os.path.join(self.folder, "tmp")
        os.makedirs(tmp, exist_ok=True)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": self.folder, "TMPDIR": tmp,
               "PIP_CONFIG_FILE": os.devnull, "PIP_NO_CACHE_DIR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
               "PYTHONNOUSERSITE": "1"}
        py = os.path.join(venv, "bin", "python")
        steps = [("make the environment", [sys.executable, "-m", "venv", venv]),
                 ("install the pinned packages", [py, "-m", "pip", *PIP_ARGS, req_file])]
        for what, argv in steps:
            code, out = (fake_step(argv, venv) if FAKE_INSTALL else run_steps(argv, env, self.alive, self.folder))
            if code != 0:
                shutil.rmtree(venv, ignore_errors=True)
                hint = (" python3's venv module does not work here; D89: no apt, so the install stops"
                        if what == "make the environment" else "")
                raise Stop(INSTALL_FAILED, f"could not {what} (exit code {code}).{hint}\n{out}")
        listing = ("import json, importlib.metadata as m; print(json.dumps({d.metadata['Name']: [d.version, "
                   "d.metadata.get('License-Expression') or d.metadata.get('License') or ''] for d in m.distributions()}))")
        code, out = fake_step([py, "-c", listing], venv) if FAKE_INSTALL else run_steps([py, "-c", listing], env, self.alive, self.folder)
        packages = json.loads(out.strip().splitlines()[-1]) if code == 0 else {}
        done = {"runtime": req.get("runtime", "litert"), "venv": spec["venv"], "python": os.path.join(spec["venv"], "bin", "python"),
                "requirements_sha256": sha256(req_file), "packages": packages,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(record, "w") as f:
            json.dump(done, f, indent=1)
        shutil.rmtree(tmp, ignore_errors=True)
        return {"built": True, "installed": done}

    def op_bye(self, req):
        return {"bye": True}


def fake_step(argv, venv):
    """--fake-install: the venv is a folder whose python is this one; pip and the listing are pretended."""
    if argv[1:3] == ["-m", "venv"]:
        os.makedirs(os.path.join(venv, "bin"), exist_ok=True)
        os.symlink(sys.executable, os.path.join(venv, "bin", "python"))
        return 0, ""
    if "-c" in argv:
        return 0, json.dumps({"ai-edge-litert": ["2.2.0", "Apache 2.0"]})
    return 0, "pretended: " + " ".join(argv[1:])


def serve(agent, lines, write):
    """Answer each request line; a request that cannot be read is answered too, never silently dropped."""
    for line in lines:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("not an object")
            rid = req.get("id")
        except ValueError:
            write({"id": None, "ok": False, "stop": LOAD_FAILED, "error": "the request is not a JSON object"})
            continue
        agent.alive = lambda done, rid=rid: write({"id": rid, "alive": True, "done": done})
        try:
            write({"id": rid, "ok": True, **agent.handle(req)})
        except Stop as e:
            write({"id": rid, "ok": False, "stop": e.stop, "error": str(e)})
        except Exception as e:                         # a fault in the agent itself: reported, the agent keeps going
            write({"id": rid, "ok": False, "stop": LOAD_FAILED, "error": f"agent fault: {type(e).__name__}: {e}"})
        if req.get("op") == "bye":
            return


def main(argv):
    global ALIVE_S
    folder = argv[1]
    opt = lambda name, default: argv[argv.index(name) + 1] if name in argv else default
    runtime = opt("--runtime", "litert")
    ALIVE_S = float(opt("--alive-s", ALIVE_S))
    FakeRuntime.ms = float(opt("--fake-ms", 0))          # tests only: how long one fake inference takes
    global FAKE_INSTALL
    FAKE_INSTALL = "--fake-install" in argv
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)

    def write(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    serve(Agent(folder, runtime), sys.stdin, write)


if __name__ == "__main__":
    main(sys.argv)
