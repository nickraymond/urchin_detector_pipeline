"""The "What runs where" page (S6 bite 4): GET /fit shows the table, POST /fit/run-all queues every
missing cell as `convert` jobs, POST /fit/run queues one cell again. Registered by app/web/server.py."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import config, modeldefs
from app.fit import convert, table
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
        return templates.TemplateResponse(request, "fit.html", {
            "cfg": cfg, "note": note, "error": error, "rows": table.table(cfg.data_dir, active),
            "cameras": [(c, convert.CAMERAS[c]) for c in table.CAMERA_ORDER], "active": len(active),
            "missing": len(table.missing(cfg.data_dir)) - len(active), "mib": table.mib, "memory_figure": table.memory_figure,
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

    @app.post("/fit/run")
    def fit_run_one(request: Request, key: str = Form(""), camera: str = Form("")):
        """Run one cell. A cell that already has a result is run again from scratch (its 8-bit model is
        made again too); one that was never tried reuses an 8-bit model another camera already made."""
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        arch = next((a for a in modeldefs.candidates() if modeldefs.arch_key(a) == key), None)
        if arch is None or camera not in convert.CAMERAS:
            return back(error=f"No such cell: {key!r} on {camera!r}.")
        jc = jobctl(cfg)
        if (key, camera) in active_cells(jc):
            return back(note=f"{key} on {convert.CAMERAS[camera]} is already waiting or running.")
        try:
            again = convert.read_fit(convert.model_dir(cfg.data_dir, arch), camera) is not None
            jc.submit("convert", convert.params(arch, camera, cfg.data_dir, cfg.tools_dir, redo=again, fit=cfg.fit))
        except JobError as e:
            return back(error=str(e))
        return back(note=f"{key} on {convert.CAMERAS[camera]} queued{' again' if again else ''}.")
