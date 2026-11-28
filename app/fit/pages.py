"""The "What runs where" page (S6 bite 4): GET /fit shows the table, POST /fit/run-all queues every
missing cell as `convert` jobs, POST /fit/run queues one cell again. Registered by app/web/server.py."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import config, modeldefs
from app.fit import convert, setup, table
from app.jobctl import JobError

WAITING = ("scheduled", "queued", "running", "pausing", "paused", "interrupted")


def register(app, templates, config_path, jobctl, dashboard) -> None:
    def active_cells(jc) -> set[tuple[str, str]]:
        return {(modeldefs.arch_key(j["params"]["arch"]), j["params"]["camera"])
                for j in jc.list() if j["kind"] == "convert" and j["state"] in WAITING}

    def back(note: str = "", error: str = "") -> RedirectResponse:
        query = urlencode({k: v for k, v in (("note", note), ("error", error)) if v})
        return RedirectResponse("/fit" + ("?" + query if query else ""), status_code=303)

    @app.get("/fit", response_class=HTMLResponse)
    def fit_page(request: Request, note: str = "", error: str = ""):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        active = active_cells(jobctl(cfg))
        rows = table.table(cfg.data_dir, active)
        return templates.TemplateResponse(request, "fit.html", {
            "cfg": cfg, "note": note, "error": error, "rows": rows, "together": table.stored_together(rows),
            "tools": setup.items(cfg.tools_dir), "outside": setup.outside(cfg.fit),
            "setup_job": jobctl(cfg).latest("tools_setup"),
            "cameras": [(c, convert.CAMERAS[c]) for c in table.CAMERA_ORDER], "active": len(active),
            "missing": len(table.missing(cfg.data_dir)) - sum(c in convert.CAMERAS for _, c in active), "mib": table.mib, "memory_figure": table.memory_figure,
            "earlier_source": table.EARLIER_SOURCE})

    @app.post("/fit/run-all")
    def fit_run_all(request: Request):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        jc = jobctl(cfg)
        active, queued = active_cells(jc), 0
        try:
            for arch, camera in table.missing(cfg.data_dir):
                if (modeldefs.arch_key(arch), camera) not in active:
                    jc.submit("convert", convert.params(arch, camera, cfg.data_dir, cfg.tools_dir, fit=cfg.fit))
                    queued += 1
        except JobError as e:
            return back(error=str(e))
        return back(note=f"{queued} cells queued; they run one at a time." if queued
                    else "Nothing to run: every cell has a result or is already waiting.")

    @app.post("/fit/setup")
    def fit_setup(request: Request):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        jc = jobctl(cfg)
        last = jc.latest("tools_setup")
        if last and last["state"] in WAITING:
            return back(note="The tools are already being set up; see Jobs.")
        try:
            jc.submit("tools_setup", {"tools_dir": str(cfg.tools_dir)})
        except JobError as e:
            return back(error=str(e))
        return back(note="Setting the tools up: this downloads several GB the first time and can take "
                         "20 minutes or more. Progress and the log are on the Jobs page.")

    @app.post("/fit/run")
    def fit_run_one(request: Request, key: str = Form(""), camera: str = Form("")):
        """Run one cell. A cell that already has a result is run again from scratch: its 8-bit model is made
        again, and since the cameras share that file, the candidate's other cells that have a result or a job under way are queued
        again behind it (QE PR 80 nit 4). One that was never tried reuses an 8-bit model already made."""
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        arch = next((a for a in modeldefs.candidates() if modeldefs.arch_key(a) == key), None)
        if arch is None or camera not in convert.STEP_LABELS:
            return back(error=f"No such cell: {key!r} on {camera!r}.")
        jc = jobctl(cfg)
        if (key, camera) in active_cells(jc):
            return back(note=f"{key} on {convert.STEP_LABELS[camera]} is already waiting or running.")
        try:
            folder = convert.model_dir(cfg.data_dir, arch)
            again = convert.read_fit(folder, camera) is not None
            # A camera with a job still waiting or running counts too: that job's result is wiped by the redo
            # behind it, so the cell is queued once more after the redo (QE PR 85). A job that then finds the
            # result on disk is skipped.
            active = active_cells(jc)
            steps = [*table.CAMERA_ORDER, *(s for s, _ in convert.SECOND.values())]   # all share the 8-bit file
            others = [c for c in steps if again and c != camera
                      and (convert.read_fit(folder, c) is not None or (key, c) in active)]
            jc.submit("convert", convert.params(arch, camera, cfg.data_dir, cfg.tools_dir, redo=again, fit=cfg.fit))
            for c in others:
                jc.submit("convert", convert.params(arch, c, cfg.data_dir, cfg.tools_dir, fit=cfg.fit))
        except JobError as e:
            return back(error=str(e))
        also = (f", and its cells on {' and '.join(convert.STEP_LABELS[c] for c in others)} with it (they share "
                "its 8-bit model)") if others else ""
        return back(note=f"{key} on {convert.STEP_LABELS[camera]} queued{' again' if again else ''}{also}.")
