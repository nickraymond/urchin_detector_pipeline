"""The "Rig log" page (S7 bite 1) and the test cards (bite 2a): GET /bench/rig-log lists the bench units from [bench] and every change the
runner made on the rig, newest first. Read-only. Registered by app/web/server.py."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import config
from app.bench import card, riglog
from app.bench import run as runs
from app.jobctl import JobError
from app.bench.board import KINDS


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
            "field_units": cfg.bench.field_count, "log_path": riglog.path(cfg.data_dir),
            "unfinished": sum(1 for r in rows if r["state"] == "not finished"),
            "undo": riglog.undo_steps(rows)})


def register_cards(app, templates, config_path, dashboard, jobctl) -> None:
    """Test cards (S7 bite 2a, D84): GET /bench/cards lists them and has the form for a new one;
    GET /bench/cards/<n> shows one; POST saves a new card or records its review. A saved card never changes."""

    def back(path: str, **query) -> RedirectResponse:
        return RedirectResponse(path + ("?" + urlencode(query) if query else ""), status_code=303)

    def cards_view(request, cfg, error: str = "", typed: dict | None = None, status: int = 200):
        defaults = {"windows": card.DEFAULT_WINDOWS, "n": card.DEFAULT_N, "metric": card.DEFAULT_METRIC,
                    "stop_rules": "\n".join(card.DEFAULT_STOP_RULES)}
        return templates.TemplateResponse(request, "bench_cards.html", {
            "cfg": cfg, "error": error, "cards": card.list_cards(cfg.data_dir), "units": cfg.bench.names(),
            "kinds": KINDS, "f": {**defaults, **(typed or {})}, "min_windows": card.MIN_WINDOWS},
            status_code=status)

    @app.get("/bench/cards", response_class=HTMLResponse)
    def cards_page(request: Request, error: str = ""):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        return cards_view(request, cfg, error)

    @app.post("/bench/cards")
    async def cards_save(request: Request):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        form = dict(await request.form())
        try:
            n = card.save(cfg.data_dir, card.build(cfg.data_dir, cfg.bench, form))
        except card.CardError as e:     # the form comes back filled in (QE PR 101 nit 3)
            return cards_view(request, cfg, str(e), {k: str(v) for k, v in form.items()}, status=400)
        return back(f"/bench/cards/{n}", note=f"Card {n} saved. It can be run once someone other than its writer has reviewed it.")

    @app.get("/bench/cards/{number}", response_class=HTMLResponse)
    def card_page(request: Request, number: int, note: str = "", error: str = ""):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        try:
            c = card.load(cfg.data_dir, number)
        except card.CardError as e:
            return HTMLResponse(str(e), status_code=404)
        jc, found = jobctl(cfg), runs.runs_for_card(cfg.data_dir, number)
        for r in found:
            jid = r["run"].get("job_id")
            try:
                r["job"] = jc.status(jid) if jid else None
            except JobError:
                r["job"] = None
        return templates.TemplateResponse(request, "bench_card.html", {
            "cfg": cfg, "c": c, "note": note, "error": error, "runs": found})

    @app.post("/bench/cards/{number}/run")
    def card_run(request: Request, number: int):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        try:
            jobctl(cfg).submit("bench", runs.params(cfg, number))
        except JobError as e:
            return back(f"/bench/cards/{number}", error=str(e))
        return back(f"/bench/cards/{number}", note="Run queued: progress and the log are on the Jobs page.")

    @app.post("/bench/cards/{number}/review")
    def card_review(request: Request, number: int, reviewed_by: str = Form("")):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        try:
            card.review(cfg.data_dir, number, reviewed_by)
        except card.CardError as e:
            return back(f"/bench/cards/{number}", error=str(e))
        return back(f"/bench/cards/{number}", note="Review recorded.")
