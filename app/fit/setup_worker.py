"""Worker of the `tools_setup` job (S6 bite 3d): `python -m app.fit.setup_worker <job.json>`. Makes each
missing tool of app/fit/setup.py with `uv` and `git`; one that is already right is skipped. Every command
and its output go to the job's log. A tool that fails stops the job with the command's last lines."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from app.fit import setup
from app.jobctl import worker


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(done.stdout, flush=True)
    if done.returncode != 0:
        last = "\n".join(done.stdout.strip().splitlines()[-6:])
        raise RuntimeError(f"`{' '.join(cmd[:4])} …` failed (exit code {done.returncode}):\n{last}")


def make_env(item: dict, uv: str) -> None:
    env = Path(item["path"])
    if not Path(uv).is_file() and not shutil.which(uv):
        raise RuntimeError(f"uv not found ({uv}). Fix: install uv with its standalone installer "
                           "(SPEC: never with Homebrew), then press the button again.")
    env.parent.mkdir(parents=True, exist_ok=True)
    (env / setup.MARK).unlink(missing_ok=True)
    if not (env / "bin" / "python").is_file():
        run([uv, "venv", "--python", item["python"], str(env)])
    spec = setup.TOOLCHAIN / item["spec"]
    what = ["-r", str(spec)] if spec.is_file() else [item["spec"]]
    run([uv, "pip", "install", "--python", str(env / "bin" / "python"), *what])
    (env / setup.MARK).write_text(setup.want(item["spec"]) + "\n")


def make_checkout(item: dict, git: str) -> None:
    path = Path(item["path"])
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        run([git, "clone", item["repo"], str(path)])
    else:
        run([git, "fetch", "origin"], cwd=path)
    run([git, "checkout", "--detach", item["commit"]], cwd=path)


def main() -> None:
    job = worker.load_job()
    todo = setup.items(Path(job["tools_dir"]))
    worker.emit("loaded", seconds=0.0)
    for i, item in enumerate(todo, 1):
        if item["ok"]:
            worker.emit("item", i=i, n=len(todo), name=item["name"], skipped=True)
            continue
        t = time.perf_counter()
        if item["kind"] == "env":
            make_env(item, job["uv"])
        else:
            make_checkout(item, job["git"])
        worker.emit("item", i=i, n=len(todo), name=item["name"], seconds=round(time.perf_counter() - t, 1),
                    counts={"made": 1})
    worker.emit("done")


if __name__ == "__main__":
    worker.run(main)
