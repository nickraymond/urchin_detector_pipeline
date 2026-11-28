"""The "Rig log" page (S7 bite 1): GET /bench/rig-log lists the bench units from [bench] and every change the
runner made on the rig, newest first. Read-only. Registered by app/web/server.py."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse

from app import config
from app.bench import riglog


def register(app, templates, config_path, dashboard) -> None:
    @app.get("/bench/rig-log", response_class=HTMLResponse)
    def rig_log_page(request: Request, show_fake: bool = False):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        rows = riglog.read(cfg.data_dir)
        fake = sum(1 for r in rows if r.get("fake"))
        if not show_fake:
            rows = [r for r in rows if not r.get("fake")]
        return templates.TemplateResponse(request, "rig_log.html", {
            "cfg": cfg, "rows": rows, "fake": fake, "show_fake": show_fake,
            "units": [cfg.bench.units[n] for n in sorted(cfg.bench.units)],
            "field_units": len(cfg.bench.field_units), "log_path": riglog.path(cfg.data_dir),
            "unfinished": sum(1 for r in rows if r["state"] == "not finished")})
