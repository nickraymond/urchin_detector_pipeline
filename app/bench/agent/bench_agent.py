"""The bench agent (S7 bite 3, D81): runs ON the bench Pi, in its own folder, started by the laptop over ssh as
`python3 bench_agent.py <folder> [--runtime litert|fake]`. Plain Python 3 and its standard library; no import from
the app (it is copied alone). numpy and the runtime are imported only when a model is loaded.

Protocol: JSON lines. The laptop writes one request per line, {"id": n, "op": ..., ...}; the agent answers each
with one line, {"id": n, "ok": true, ...} or {"id": n, "ok": false, "stop": <stop rule>, "error": <words>}.
Ops: hello, info, load, time, reset, bye. Model files are copied into <folder>/models/ by the laptop before
"load"; the agent only checks their sha256 (a mismatch is the "readback_differs" stop rule, D84).

Timing (D83): one seeded synthetic input at the model's input size; 2 warm-ups; time.perf_counter_ns() around
each invoke of the window; peak memory is the process's VmHWM. Threads are fixed per load and recorded.
The cameras (n6, ae3) are bites 4 and 5: asked for one, the agent says so.
"""

import hashlib
import json
import os
import platform
import re
import sys
import time

VERSION = "1"
WARMUPS = 2
SEED = 0
NO_ANSWER, READBACK, LOAD_FAILED, NOT_LOADED, NOT_BUILT = (
    "no_answer", "readback_differs", "load_failed", "not_loaded", "not_built")


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
    """No runtime and no numpy: invoke does nothing. For the laptop's tests of the protocol and the transport."""
    name = "fake"

    def open(self, path, threads):
        return {"path": path}

    def invoke(self, model):
        pass

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


class Agent:
    def __init__(self, folder, runtime="litert"):
        self.folder = os.path.abspath(folder)
        self.models_dir = os.path.join(self.folder, "models")
        self.runtime = RUNTIMES[runtime]()
        self.loaded = {}          # name -> (runtime model, threads)

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
        for _ in range(WARMUPS):
            self.runtime.invoke(model)
        each = []
        for _ in range(n):
            t = time.perf_counter_ns()
            self.runtime.invoke(model)
            each.append((time.perf_counter_ns() - t) / 1e6)
        return {"n": n, "warmups": WARMUPS, "threads": threads, "ms_mean": sum(each) / n, "ms_min": min(each),
                "ms_max": max(each), "vm_hwm": proc_kib("/proc/self/status", "VmHWM"),
                "mem_free": proc_kib("/proc/meminfo", "MemAvailable")}

    def op_reset(self, req):
        self.loaded = {}
        return {"reset": True}

    def op_bye(self, req):
        return {"bye": True}


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
        try:
            write({"id": rid, "ok": True, **agent.handle(req)})
        except Stop as e:
            write({"id": rid, "ok": False, "stop": e.stop, "error": str(e)})
        except Exception as e:                         # a fault in the agent itself: reported, the agent keeps going
            write({"id": rid, "ok": False, "stop": LOAD_FAILED, "error": f"agent fault: {type(e).__name__}: {e}"})
        if req.get("op") == "bye":
            return


def main(argv):
    folder = argv[1]
    runtime = argv[argv.index("--runtime") + 1] if "--runtime" in argv else "litert"
    os.makedirs(os.path.join(folder, "models"), exist_ok=True)

    def write(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    serve(Agent(folder, runtime), sys.stdin, write)


if __name__ == "__main__":
    main(sys.argv)
