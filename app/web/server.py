"""The labeling tool's web pages. Start with `uv run app`; it listens on 127.0.0.1 only
(SPEC constraint 1) and opens the dashboard in the default browser.

The settings file is re-read on every page load, so plugging in the SSD or fixing
config.local.toml only needs a browser reload, not a restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import uvicorn
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import Response, FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import config, db, project
from app.train import job as train_job
from app.web import datasets_pages, train_pages
from app.jobctl import JobControl, JobError
from app.jobctl import limits as job_limits
from app.jobctl import awake, schedule
from app.jobctl.runner import GUARD_REASON
from app.datasets import worker as datasets_worker
from app.export import worker as export_worker
from app.fit import convert as fit_convert
from app.fit import pages as fit_pages
from app.bench import pages as bench_pages
from app.fit import setup as fit_setup
from app.sam import add, classes
from app.review import state as review_state
from app.stages import clips, extract, ingest, label, review, review_labels

HOST = "127.0.0.1"  # never 0.0.0.0: images never leave the laptop (SPEC constraint 1)
# Host and origin rule (DESIGN D14): pages answer only to these host names, and a POST is
# accepted only from a page this app served. Stops other web pages in the browser (and DNS
# rebinding) from reading pages or changing projects.
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
WEB = Path(__file__).parent
templates = Jinja2Templates(directory=WEB / "templates")


def static_url(name: str) -> str:
    """/static/<name>?v=<content hash>: the address changes whenever the file does, so the
    browser never keeps using an old saved copy of a script or the stylesheet (Nick 2026-11-25)."""
    data = (WEB / "static" / name).read_bytes()
    return f"/static/{name}?v={hashlib.sha1(data).hexdigest()[:8]}"


templates.env.globals["static"] = static_url
JOB_NAMES = {"sam": "SAM labeling", "extract": "Extraction", "playback": "Playback copy",
             "export": "Export", "sam_add": "Add a missed object",
             "freeze": "Freeze a dataset version", "train": "Training"}
AWAKE_CMD: list[str] | None = None   # tests: a fake keep-awake command (D38); None = the real one
# A duration as the pages show it: "7 min", or "2.5 h" from 90 min on.
templates.env.globals["minutes"] = lambda s: (
    f"{max(1, math.ceil(s / 60))} min" if s < 5400 else f"{s / 3600:.1f} h")


def create_app(config_path: Path = config.DEFAULT_PATH) -> FastAPI:
    app = FastAPI(title="Urchin labeling tool", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        response = None
        if request.method not in ("GET", "HEAD"):
            source = request.headers.get("origin") or request.headers.get("referer") or ""
            parts = urlsplit(source)
            if parts.scheme != "http" or parts.netloc != request.headers.get("host"):
                response = PlainTextResponse(
                    "Refused: this action can only come from the app's own pages "
                    f"(got origin {source or 'none'!r}).", status_code=403)
        response = response or await call_next(request)
        # Other web pages in the browser may not embed anything this app serves, e.g. a
        # thumbnail or frame in an <img> (D14).
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"  # ask the app before reusing a copy
        return response

    # Added last so it runs first: a request with a foreign Host never reaches a page.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    # --- Job control (S4, D32): one JobControl per data folder, made at app start ---
    controls: dict[str, JobControl] = {}

    def jobctl(cfg) -> JobControl:
        """The job module for this data folder; creating it marks jobs left running by a quit
        or crash as interrupted (D34). Only called when cfg.ok (data_dir exists, D13)."""
        key = str(cfg.data_dir)
        if key not in controls:
            # not started yet: every kind is registered first, so a job that is queued or due at
            # app start never meets a runner that does not know its kind (TRACKER S8 Fixes)
            jc = JobControl(cfg.data_dir / "jobs", start=False, awake_cmd=AWAKE_CMD)
            jc.register(label.sam_kind())
            jc.register(label.add_kind())
            jc.register(export_worker.export_kind())
            jc.register(datasets_worker.freeze_kind())
            jc.register(train_job.train_kind())
            jc.register(extract.extract_kind())
            jc.register(clips.playback_kind())
            jc.register(fit_convert.convert_kind())   # Build 2 (S6): one model x one camera
            jc.register(fit_setup.setup_kind())       # Build 2 (S6): the tool environments
            jc.start()
            controls[key] = jc
        return controls[key]

    def close_jobs() -> None:
        """Release the job module (tests that start a second app in the same process)."""
        for jc in controls.values():
            jc.close()
        controls.clear()

    app.state.jobctl = jobctl
    app.state.close_jobs = close_jobs
    first = config.load(config_path)
    if first.ok:
        jobctl(first)

    def finish_time(seconds: float, start: str | None = None) -> str:
        """Clock time `seconds` after `start` (ISO; default now), as the page shows it (e.g.
        "6:42 PM", or with the weekday when it is not today)."""
        from datetime import datetime, timedelta
        at = (schedule.parse(start) or datetime.now()) + timedelta(seconds=seconds)
        return schedule.show(schedule.iso(at))

    def job_view(job: dict | None) -> dict:
        """A job record as pages and status calls show it; {} when there is none."""
        if not job:
            return {}
        total = job.get("total") or 0
        eta = job.get("eta_s") if job["state"] in ("running", "pausing") else None
        line = job.get("message") or ""
        if eta and job["state"] == "running":   # bite 6: time, finish, then what
            line = (f"about {templates.env.globals['minutes'](eta)} left, done by "
                    f"{finish_time(eta)} · {line}")
        params = job.get("params") or {}
        guard = job["state"] == "paused" and (job.get("stop_reason") or "").startswith(
            GUARD_REASON)
        return {**job, "line": line, "project_name": Path(job["project"]).name if job.get("project") else "",
                "what": JOB_NAMES.get(job["kind"], job["kind"]) + (
                    " (preview)" if params.get("preview") else
                    " (estimate)" if params.get("estimate") else ""),
                "item": ("frame" if job["kind"] in ("sam", "extract", "export", "freeze")
                         else "epoch" if job["kind"] == "train" else "item"),
                "guard_paused": guard,   # D36c: resumes by itself at twice the limit
                "guard_resume_gb": f"{2 * float((job.get('limits') or {}).get('memory_guard_gb') or 0):g}",
                "percent": round(100 * job["done"] / total, 1) if total else 0.0,
                "active": job["state"] in ("queued", "running", "pausing"),
                "start_show": schedule.show(job.get("start_at")),
                # the summary's counts, urchins first (the number the morning summary is for)
                "found": sorted(((job.get("summary") or {}).get("counts") or {}).items(),
                                key=lambda kv: ("urchin" not in kv[0], -kv[1], kv[0])),
                "stop_show": schedule.show(job.get("stop_by"))}

    def read_schedule(start: str, stop: str):
        """(start_at, stop_by, problem) from the two optional clock-time fields (D37)."""
        try:
            return (*schedule.plan(start, stop), "")
        except schedule.ScheduleError as e:
            return None, None, str(e)

    def waiting_jobs(cfg, project_name: str = "") -> list[dict]:
        """Jobs waiting for Nick (bite 6b item 4): paused, pausing or interrupted; optionally
        of one project only."""
        return [v for v in (job_view(j) for j in jobctl(cfg).list())
                if v["state"] in ("pausing", "paused", "interrupted")
                and (not project_name or v["project_name"] == project_name)]

    def dashboard(request: Request, error: str = "", name: str = "", status: int = 200):
        cfg = config.load(config_path)
        projects = project.list_projects(cfg.data_dir) if cfg.ok else []
        return templates.TemplateResponse(request, "dashboard.html", {
            "cfg": cfg, "projects": projects, "error": error, "name": name,
            "waiting": waiting_jobs(cfg) if cfg.ok else [], "name_help": project.NAME_HELP,
            "last": job_view(jobctl(cfg).last_summary()) if cfg.ok else {}},
            status_code=status)

    def awake_view(jc) -> dict:
        """Keep-awake state for the pages (D38)."""
        return {"pid": jc.awake.pid(), "error": jc.awake.error, "battery": awake.on_battery()}

    @app.get("/jobs/{job_id}/log")
    def job_log(job_id: int):
        """The job's log as plain text (reads only)."""
        cfg = config.load(config_path)
        if not cfg.ok:
            return PlainTextResponse("Setup problem; open the dashboard.", status_code=503)
        p = jobctl(cfg).root / f"{job_id}.log"
        if not p.is_file():
            return PlainTextResponse(f"No log for job {job_id}.", status_code=404)
        return PlainTextResponse(p.read_text(errors="replace"))

    # --- Live fragments (bite 6b item 6): pages poll these and swap only what changed ---
    LIVE = {
        "panel": templates.env.from_string(
            '{% from "_job.html" import job_panel %}{{ job_panel(j, back, compact) }}'),
        "banner": templates.env.from_string(
            '{% from "_job.html" import waiting_banner %}{{ waiting_banner(jobs, back) }}')}

    def live_html(text: str) -> HTMLResponse:
        return HTMLResponse(text, headers={"Cache-Control": "no-store"})

    @app.get("/live/jobs", response_class=HTMLResponse)
    def live_jobs(request: Request):
        cfg = config.load(config_path)
        if not cfg.ok:
            return live_html("")
        return templates.TemplateResponse(request, "_jobs_table.html", {
            "jobs": [job_view(j) for j in jobctl(cfg).list()]},
            headers={"Cache-Control": "no-store"})

    @app.get("/live/banner", response_class=HTMLResponse)
    def live_banner(project: str = "", back: str = "/jobs"):
        cfg = config.load(config_path)
        if not cfg.ok:
            return live_html("")
        return live_html(LIVE["banner"].render(jobs=waiting_jobs(cfg, project),
                                               back=safe_back(back)))

    @app.get("/p/{name}/live/{kind}", response_class=HTMLResponse)
    def live_panel(request: Request, name: str, kind: str, back: str = "", compact: str = ""):
        """The latest job of this kind in this project, as the pages show it."""
        if kind not in ("sam", "extract", "export", "freeze", "train"):
            return PlainTextResponse(f"No job kind {kind!r}.", status_code=404)
        return with_project(request, name, lambda cfg, info, conn: live_html(
            LIVE["panel"].render(j=job_view(jobctl(cfg).latest(kind, str(info["folder"]))),
                                 back=safe_back(back or f"/p/{name}"), compact=bool(compact))))

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request, note: str = "", error: str = ""):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        jc = jobctl(cfg)
        return templates.TemplateResponse(request, "jobs.html", {
            "cfg": cfg, "note": note, "error": error, "jobs": [job_view(j) for j in jc.list()],
            "limits": job_limits.load(jc.root), "devices": job_limits.DEVICES,
            "cores": os.cpu_count() or 1, "free_gb": round(job_limits.free_gb(), 1),
            "awake": awake_view(jc), "waiting": waiting_jobs(cfg)})

    @app.post("/jobs/settings")
    def jobs_settings(request: Request, device: str = Form("auto"), threads: str = Form(""),
                      low_priority: str = Form(""), pause_s: str = Form("0"),
                      memory_guard: str = Form(""), memory_guard_gb: str = Form("")):
        """Save the limits (D36); they apply to jobs that start from now on."""
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        try:
            job_limits.save(jobctl(cfg).root, {
                "device": device, "threads": threads, "low_priority": low_priority,
                "pause_s": pause_s, "memory_guard": memory_guard,
                "memory_guard_gb": memory_guard_gb})
        except job_limits.LimitsError as e:
            return RedirectResponse("/jobs?" + urlencode({"error": str(e)}), status_code=303)
        return RedirectResponse("/jobs?" + urlencode(
            {"note": "Saved. Jobs that start from now on use these limits; a running job "
                     "keeps the ones it started with."}), status_code=303)

    def safe_back(back: str) -> str:
        """Only a path on this app: "/..." but not "//host"."""
        return back if back.startswith("/") and not back.startswith("//") else "/jobs"

    @app.post("/jobs/{job_id}/{action}")
    def job_action(request: Request, job_id: int, action: str, back: str = Form("/jobs")):
        """Pause / Resume / Cancel any job (D34); back to the page the button was on."""
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        do = {"pause": "pause", "resume": "resume", "cancel": "cancel",
              "start-now": "start_now"}.get(action)
        if do is None:
            return PlainTextResponse(f"No job action {action!r}.", status_code=404)
        back, note = safe_back(back), ""
        try:
            getattr(jobctl(cfg), do)(job_id)
        except JobError as e:
            note = str(e)
        sep = "&" if "?" in back else "?"
        return RedirectResponse(back + (sep + urlencode({"note": note}) if note else ""),
                                status_code=303)

    fit_pages.register(app, templates, config_path, jobctl, dashboard)   # Build 2: "What runs where"
    bench_pages.register(app, templates, config_path, dashboard)          # Build 2 (S7): the rig log
    bench_pages.register_cards(app, templates, config_path, dashboard)    # Build 2 (S7): test cards

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return dashboard(request)

    @app.post("/projects")
    def new_project(request: Request, name: str = Form("")):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        name = name.strip()
        try:
            project.create(cfg.data_dir, name)
        except project.ProjectError as e:
            return dashboard(request, error=str(e), name=name, status=400)
        except OSError as e:
            return dashboard(request, error=f"Could not create the project folder: {e}. "
                             "Check that the SSD is plugged in and not full.", name=name,
                             status=500)
        return RedirectResponse(f"/p/{name}", status_code=303)

    def with_project(request: Request, name: str, work):
        """Open the project, run work(cfg, info, conn), always close the database.
        Settings or project problems go back to the dashboard with the message."""
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        try:
            info = project.open_project(cfg.data_dir, name)
            conn = db.connect(info["folder"] / "state.sqlite")
        except (project.ProjectError, OSError, sqlite3.Error) as e:
            return dashboard(request, error=str(e), status=404)
        try:
            return work(cfg, info, conn)
        finally:
            conn.close()

    def stages_with_labels(info, conn) -> list[dict]:
        """The stage table, with SAM labeling counted from the results on disk (bite 6c, Nick:
        a finished run still said "not started"): a kept frame counts once it has a results
        file. "done" = every kept frame has one; a class added later still shows on the
        Labeling page's Run plan."""
        stages = db.stage_statuses(conn)
        files = label.frames_to_label(conn)
        n = len(label.labeled_frames(info["folder"], files))
        for s in stages:
            if s["stage"] == "labeling" and files and n:
                s.update(state="done" if n == len(files) else "in progress",
                         counts=f"{n} of {len(files)} frames labeled")
        return stages

    @app.get("/p/{name}", response_class=HTMLResponse)
    def project_page(request: Request, name: str):
        return with_project(request, name, lambda cfg, info, conn: templates.TemplateResponse(
            request, "project.html", {
                "cfg": cfg, "info": info, "stages": stages_with_labels(info, conn),
                "waiting": waiting_jobs(cfg, info["name"]),
                # bite 6b item 8: each job stage row shows its latest job, live
                "jobs": {k: job_view(jobctl(cfg).latest(k, str(info["folder"])))
                         for k in ("sam", "extract")}}))

    def ingest_url(name: str, folder: str) -> str:
        return f"/p/{name}/ingest?" + urlencode({"folder": folder})

    @app.get("/p/{name}/ingest", response_class=HTMLResponse)
    def ingest_page(request: Request, name: str, folder: str = "", note: str = ""):
        def work(cfg, info, conn):
            start = folder.strip() or next((str(r) for r in cfg.video_roots if r.is_dir()), "")
            here = Path(start).expanduser() if start else None
            error, subfolders, videos = "", [], []
            if here is None:
                error = "No video folder found. Paste a folder path below."
            elif not here.is_dir():
                error = f"Folder not found: {here}. Check the path, or plug in the SSD."
            else:
                subfolders = sorted(p for p in here.iterdir()
                                    if p.is_dir() and not p.name.startswith("."))
                try:
                    videos = ingest.scan(here)
                except ingest.IngestError as e:
                    error = str(e)
            for v in videos:
                ingest.thumbnail(v, cfg.data_dir / "thumbs")
            stills = ingest.find_stills(here) if here is not None and here.is_dir() else []
            have = ingest.added_stills(conn)
            keys = ingest.added_keys(conn)
            return templates.TemplateResponse(request, "ingest.html", {
                "cfg": cfg, "info": info, "here": here, "error": error,
                "subfolders": subfolders, "videos": videos, "added": ingest.added(conn),
                "stills": len(stills), "stills_new": sum(
                    1 for p in stills if (p.name, p.stat().st_size) not in have),
                "stills_added": len(have), "max_stills": ingest.MAX_STILLS, "note": note,
                "added_paths": set(keys.values()),
                # a copy of this video from another folder is already in the project (D13)
                "added_elsewhere": {str(v.path): keys[(v.path.name, v.size)] for v in videos
                                    if (v.path.name, v.size) in keys
                                    and keys[(v.path.name, v.size)] != str(v.path)}})
        return with_project(request, name, work)

    @app.get("/thumbs/{code}.jpg")
    def thumb(code: str):
        """Video thumbnails made by the ingest page; only names like 3fa9c1.jpg, only from
        <data_dir>/thumbs/ (no path from the request reaches the disk)."""
        cfg = config.load(config_path)
        path = cfg.data_dir / "thumbs" / f"{code}.jpg" if cfg.ok else None
        if not ingest.THUMB_CODE.match(code) or path is None or not path.is_file():
            return PlainTextResponse("no thumbnail", status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/p/{name}/ingest/add")
    def ingest_add(request: Request, name: str, folder: str = Form(""),
                   paths: list[str] = Form([])):
        def work(cfg, info, conn):
            # Only videos the scan of this folder found can be added, never an arbitrary path.
            picked = set(paths)
            try:
                videos = ingest.scan(Path(folder))
            except ingest.IngestError:
                videos = []
            ingest.add(conn, [v for v in videos if str(v.path) in picked])
            return RedirectResponse(ingest_url(name, folder), status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/ingest/stills")
    def ingest_stills(request: Request, name: str, folder: str = Form("")):
        """Copy the still images under this folder into the project (bite 5). Only a folder
        that exists; the images are only read."""
        def work(cfg, info, conn):
            here = Path(folder)
            if not folder or not here.is_dir():
                note = f"Folder not found: {folder}."
            else:
                try:
                    n = ingest.add_stills(conn, here, extract.frames_dir(info["folder"]))
                    def many(k, one, more):  # "1 was", "2 were" (QE nit, PR 16)
                        return f"{n[k]} {one if n[k] == 1 else more}"
                    note = (f"Added {many('added', 'still image', 'still images')}" +
                            (f"; {many('skipped', 'was', 'were')} already in the project"
                             if n["skipped"] else "") +
                            (f"; {many('unreadable', 'could not be read and was',
                                       'could not be read and were')} left out"
                             if n["unreadable"] else "") + ". They are in Review frames.")
                except ingest.IngestError as e:
                    note = str(e)
            return RedirectResponse(ingest_url(name, folder) + "&" + urlencode({"note": note}),
                                    status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/ingest/remove")
    def ingest_remove(request: Request, name: str, folder: str = Form(""), path: str = Form("")):
        def work(cfg, info, conn):
            ingest.remove(conn, path)
            return RedirectResponse(ingest_url(name, folder), status_code=303)
        return with_project(request, name, work)

    def playback_jobs(cfg, info) -> dict[str, dict]:
        """The newest playback job per video code of this project."""
        out: dict[str, dict] = {}
        for j in jobctl(cfg).list():   # newest first
            if j["kind"] == "playback" and j["project"] == str(info["folder"]):
                out.setdefault(j["params"].get("code"), job_view(j))
        return out

    strip_file = re.compile(r"^\d{4}\.jpg$")

    def clip_status(cfg, info, videos) -> dict:
        found = playback_jobs(cfg, info)
        out = {}
        for v in videos:
            code = v["path_code"]
            st = found.get(code, {})
            out[code] = {**st, "ready": clips.ready(info["folder"], code)}
        return out

    @app.get("/p/{name}/clips", response_class=HTMLResponse)
    def clips_page(request: Request, name: str, v: str = ""):
        def work(cfg, info, conn):
            videos = ingest.added(conn)
            current = next((r for r in videos if r["path_code"] == v), videos[0] if videos else None)
            status = clip_status(cfg, info, videos)
            strip = (clips.strip_frames(info["folder"], current["path_code"])
                     if current and status[current["path_code"]]["ready"] else [])
            every = info["settings"]["seconds_between_samples"]
            try:
                all_clips, clip_error = clips.load_clips(info["folder"]), ""
            except clips.ClipError as e:
                all_clips, clip_error = [], str(e)
            mine = [c for c in all_clips if current and c["video"] == current["path"]]
            return templates.TemplateResponse(request, "clips.html", {
                "cfg": cfg, "info": info, "videos": videos, "current": current,
                "clips": mine, "clip_error": clip_error, "every": every,
                "cap": info["settings"]["frame_cap"],
                "summary": clips.summary(videos, all_clips, every),
                "status": status, "strip": [p.name for p in strip],
                # videos with no copy and no job queued or running: what the button starts
                "idle": sum(1 for st in status.values() if not st["ready"]
                            and st.get("state") not in ("queued", "running")),
                "strip_every": clips.STRIP_EVERY_S})
        return with_project(request, name, work)

    @app.post("/p/{name}/clips/prepare")
    def clips_prepare(request: Request, name: str, v: str = Form("")):
        """Queue a playback copy for every video that has none yet, in order. A POST, so only
        the app's own page can start this CPU work (D14)."""
        def work(cfg, info, conn):
            found = playback_jobs(cfg, info)
            for row in ingest.added(conn):
                code = row["path_code"]
                if not clips.ready(info["folder"], code) and \
                        not found.get(code, {}).get("active"):
                    jobctl(cfg).submit("playback", clips.playback_params(info["folder"], row),
                                       project=str(info["folder"]))
            return RedirectResponse(f"/p/{name}/clips?" + urlencode({"v": v}), status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/clips/save")
    async def clips_save(request: Request, name: str):
        """Replace one video's clips: JSON {video: <path code>, clips: [{start_s, end_s, note}]}.
        Answers the checked clips and new counts, or 400 with a message to show."""
        try:
            body = await request.json()
            code, new = str(body["video"]), list(body["clips"])
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"error": "Bad request: expected {video, clips}."}, 400)

        def work(cfg, info, conn):
            videos = ingest.added(conn)
            video = next((r for r in videos if r["path_code"] == code), None)
            if video is None:
                return JSONResponse({"error": "That video is not in this project."}, 404)
            try:
                saved = clips.save_video_clips(info["folder"], video["path"],
                                               video["duration_s"], new)
                total = clips.summary(videos, clips.load_clips(info["folder"]),
                                      info["settings"]["seconds_between_samples"])
            except clips.ClipError as e:
                return JSONResponse({"error": str(e)}, 400)
            if total["clips"]:
                db.set_stage(conn, "clips", "done", f"{total['clips']} clip"
                             f"{'s' * (total['clips'] != 1)}, "
                             f"{int(total['marked_s'] // 60)}:{int(total['marked_s'] % 60):02d} marked")
            else:
                db.set_stage(conn, "clips", "not started")
            return JSONResponse({"clips": saved, "summary": total})
        return with_project(request, name, work)

    # --- Extract frames (bite 3a) ---

    def extract_plan(info, conn, every: float, cap: int):
        videos = ingest.added(conn)
        try:
            all_clips = clips.load_clips(info["folder"])
        except clips.ClipError:
            all_clips = []
        return extract.sample_plan(videos, all_clips, every, cap)

    def check_settings(every: float, cap: int) -> str:
        if not (math.isfinite(every) and 0.1 <= every <= 600):
            return "Seconds between frames must be between 0.1 and 600."
        if not 1 <= cap <= 100_000:
            return "The frame cap must be between 1 and 100000."
        return ""

    def filter_settings(info) -> tuple[float, int]:
        st = {**project.DEFAULT_SETTINGS, **info["settings"]}  # older projects lack blur_percent
        return st["blur_percent"], st["duplicate_distance"]

    def check_filters(blur: str, dup: str) -> tuple[str, float, int]:
        """(problem, blur, dup): the slider values as numbers, or a message (400, not 422)."""
        try:
            b, d = float(blur), float(dup)
        except ValueError:
            return "The filter values must be numbers.", 0.0, 0
        if not (math.isfinite(b) and 0 <= b <= 50):
            return "The blur filter must be between 0 and 50 %.", 0.0, 0
        if not (d.is_integer() and 0 <= d <= 64):
            return "The duplicate filter must be a whole number from 0 to 64.", 0.0, 0
        return "", b, int(d)

    def planned_names(samples) -> set[str]:
        return {extract.frame_name(s["video"]["path"], s["video"]["path_code"], s["t"])
                for s in samples}

    def current_plan(info, conn) -> set[str]:
        """File names of the plan for the saved settings (D20: counts add up to it)."""
        st = info["settings"]
        samples, _ = extract_plan(info, conn, st["seconds_between_samples"], st["frame_cap"])
        return planned_names(samples)

    def extract_job(cfg, info) -> dict:
        return job_view(jobctl(cfg).latest("extract", str(info["folder"])))

    def extract_busy(cfg, info) -> bool:
        return bool(extract_job(cfg, info).get("active"))

    @app.get("/p/{name}/extract", response_class=HTMLResponse)
    def extract_page(request: Request, name: str, error: str = "", note: str = ""):
        def work(cfg, info, conn):
            st = info["settings"]
            blur, dup = filter_settings(info)
            samples, live = extract_plan(info, conn, st["seconds_between_samples"], st["frame_cap"])
            # every kept frame, for the viewer's arrows; the grid shows the first 12
            # kept frames, also by hand (D20); still images are not extracted (bite 5)
            frames = [f for f in review.frames(conn) if f["t"] is not None]
            return templates.TemplateResponse(request, "extract.html", {
                "cfg": cfg, "info": info, "every": st["seconds_between_samples"],
                "cap": st["frame_cap"], "live": live, "planned": len(samples),
                "videos": len(ingest.added(conn)), "counts": extract.counts(conn, planned_names(samples)),
                "job": extract_job(cfg, info), "frames": frames, "battery": awake.on_battery(),
                "waiting": waiting_jobs(cfg, info["name"]),
                "blur": blur, "dup": dup, "error": error, "note": note,
                "stage": next(s for s in db.stage_statuses(conn) if s["stage"] == "extract")})
        return with_project(request, name, work)

    @app.get("/p/{name}/extract/filter")
    def extract_filter_preview(request: Request, name: str, blur: str = "10", dup: str = "4"):
        """What Apply would do with these slider values: counts and examples (reads only)."""
        def work(cfg, info, conn):
            problem, b, d = check_filters(blur, dup)
            if problem:
                return JSONResponse({"error": problem}, 400)
            return JSONResponse(extract.preview(conn, b, d, planned=current_plan(info, conn)))
        return with_project(request, name, work)

    @app.post("/p/{name}/extract/filter")
    def extract_filter_apply(request: Request, name: str, blur: str = Form("10"),
                             dup: str = Form("4")):
        """Apply the filters: move files between frames/ and dropped/, save the values."""
        def work(cfg, info, conn):
            problem, b, d = check_filters(blur, dup)
            if not problem and extract_busy(cfg, info):
                problem = "Extraction is running; apply the filters when it has finished."
            if not problem:
                project.update_settings(info["folder"], blur_percent=b, duplicate_distance=d)
                extract.apply_filters(conn, info["folder"], b, d)
            return RedirectResponse(f"/p/{name}/extract" + (
                "?" + urlencode({"error": problem}) if problem else ""), status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/extract/plan")
    def extract_plan_json(request: Request, name: str, every: float = 2.0, cap: int = 300):
        """The live count for the page while settings are typed (reads only)."""
        def work(cfg, info, conn):
            problem = check_settings(every, cap)
            if problem:
                return JSONResponse({"error": problem}, 400)
            samples, live = extract_plan(info, conn, every, cap)
            return JSONResponse({"live": live, "planned": len(samples)})
        return with_project(request, name, work)

    @app.post("/p/{name}/extract/start")
    def extract_start(request: Request, name: str, every: float = Form(2.0),
                      cap: int = Form(300), start: str = Form(""), stop: str = Form(""),
                      keep_awake: str = Form("")):
        def work(cfg, info, conn):
            problem = check_settings(every, cap)
            start_at, stop_by, when = read_schedule(start, stop)
            problem = problem or when
            if not problem and extract_job(cfg, info).get("state") == "scheduled":
                problem = "An extraction is scheduled; use its Start now or Cancel first."
            if problem:
                return RedirectResponse(f"/p/{name}/extract?" + urlencode({"error": problem}),
                                        status_code=303)
            project.update_settings(info["folder"], seconds_between_samples=every, frame_cap=cap)
            if extract_busy(cfg, info):
                return RedirectResponse(f"/p/{name}/extract?" + urlencode(
                    {"error": "Extraction is already queued or running."}), status_code=303)
            samples, _ = extract_plan(info, conn, every, cap)
            jc = jobctl(cfg)
            old = extract_job(cfg, info)
            if old.get("state") in ("paused", "interrupted"):   # the new plan replaces it
                try:
                    jc.cancel(old["id"])
                except JobError:
                    pass
            jc.submit("extract", extract.job_params(info["folder"], samples),
                      project=str(info["folder"]), start_at=start_at, stop_by=stop_by,
                      keep_awake=bool(keep_awake))
            return RedirectResponse(f"/p/{name}/extract", status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/extract/status")
    def extract_status(request: Request, name: str):
        return with_project(request, name, lambda cfg, info, conn: JSONResponse({
            "job": extract_job(cfg, info),
            "counts": extract.counts(conn, current_plan(info, conn))}))

    # --- Review frames (bite 4, DESIGN D20) ---

    @app.get("/p/{name}/review", response_class=HTMLResponse)
    def review_page(request: Request, name: str, dropped: int = 0):
        def work(cfg, info, conn):
            st = info["settings"]
            samples, _ = extract_plan(info, conn, st["seconds_between_samples"], st["frame_cap"])
            return templates.TemplateResponse(request, "review.html", {
                "cfg": cfg, "info": info, "dropped": bool(dropped),
                "frames": review.frames(conn, bool(dropped)),
                "counts": extract.counts(conn, planned_names(samples)),
                "stage": next(s for s in db.stage_statuses(conn) if s["stage"] == "review"),
                "busy": extract_busy(cfg, info)})
        return with_project(request, name, work)

    @app.post("/p/{name}/review/act")
    async def review_act(request: Request, name: str):
        """JSON {file, key: "D" | "K" | "Z"}: reject, keep, or undo the last action
        (file is ignored for Z). Answers {file, status, label, changed} or {error}."""
        try:
            body = await request.json()
            file, key = str(body.get("file", "")), str(body["key"])
        except (ValueError, KeyError, TypeError, AttributeError):
            return JSONResponse({"error": "Bad request: expected {file, key}."}, 400)

        def work(cfg, info, conn):
            if extract_busy(cfg, info):
                return JSONResponse({"error": "Extraction is running; review when it has "
                                              "finished."}, 409)
            try:
                if key == "Z":
                    done = review.undo(conn, info["folder"]) or {"error": "Nothing to undo."}
                else:
                    done = review.act(conn, info["folder"], file, key)
            except review.ReviewError as e:
                return JSONResponse({"error": str(e), "summary": review.summary(conn)}, 400)
            return JSONResponse({**done, "summary": review.summary(conn)})
        return with_project(request, name, work)

    @app.post("/p/{name}/review/finish")
    def review_finish(request: Request, name: str, state: str = Form("")):
        """The Done and Skip buttons (review page and dashboard)."""
        def work(cfg, info, conn):
            try:
                review.finish(conn, state)
            except review.ReviewError as e:
                return PlainTextResponse(str(e), status_code=400)
            return RedirectResponse(f"/p/{name}", status_code=303)
        return with_project(request, name, work)

    # --- SAM labeling (S3): classes.yaml (bite 1, D25) ---

    def labeling_page_for(request, cfg, info, conn, text=None, errors=None, note="",
                          status=200, n_labeled: int = 48):
        cls = classes.read(info["folder"])
        if text is not None:  # a refused save: show what was typed, with its problems
            cls = {**cls, "text": text, "errors": errors or []}
        data = cls["data"]
        rows = [{**c, "colour": classes.colour(c["id"])} for c in data["classes"]] if data else []
        ids = [c["id"] for c in rows] + list(cls["used"]) + [0]
        units = classes.run_units(data["classes"]) if data else []
        prev = label.read_preview(info["folder"])
        shown = []
        if prev and data:
            for f in prev["files"]:
                ev = next((e for e in prev["frames"] if e["file"] == f), {})
                try:  # an unreadable sam_raw file: say so for that frame, keep the page (QE PR 22)
                    c, problem = label.frame_counts(info["folder"], f, units, data["settings"]), ""
                except ValueError as e:
                    c, problem = {}, str(e)
                shown.append({"file": f, "counts": c, "seconds": ev.get("seconds"),
                              "problem": problem})
        colours = {u["key"]: label.unit_colour(u) for u in units}
        legend_units = [{"key": u["key"], "colour": colours[u["key"]], "kind": u["kind"],
                         "untagged": bool(u["group"])} for u in units]
        legend_frames = [{"file": f["file"], "caption": f["file"] + (
                              f" · {f['seconds']} s" if f["seconds"] else ""),
                          "counts": {k: c["masks"] for k, c in f["counts"].items()}}
                         for f in shown if not f["problem"]]
        files = label.frames_to_label(conn)
        estimate = label.read_estimate(info["folder"])
        try:
            plan = (label.run_plan(info["folder"], files, units, data["settings"], prev,
                                   estimate)
                    if data else None)
        except ValueError:
            plan = None
        return templates.TemplateResponse(request, "labeling.html", {
            "cfg": cfg, "info": info, "cls": cls, "rows": rows, "note": note, "plan": plan,
            "units": units, "preview": prev, "shown": shown,
            "colours": colours, "legend_units": legend_units, "legend_frames": legend_frames,
            "form": {"settings": data["settings"], "classes": data["classes"],
                     "used": {str(k): v for k, v in cls["used"].items()},
                     "next_id": max(ids) + 1, "kinds": list(classes.KINDS),
                     "models": list(classes.MODELS), "untagged": classes.UNTAGGED_COLOUR,
                     "palette": classes.PALETTE,
                     "save": f"/p/{info['name']}/labeling/classes/form"} if data else None,
            "job": sam_job(cfg, info), "estimate": estimate, "battery": awake.on_battery(),
            "waiting": waiting_jobs(cfg, info["name"]),
            "finish": (finish_time(plan["estimate_s"], sam_job(cfg, info).get("start_at")
                                   if sam_job(cfg, info).get("state") == "scheduled" else None)
                       if plan and plan.get("estimate_s") else None),
            "n_label": len(files),
            "labeled": labeled_view(info, conn, n_labeled),
            "next_id": max(ids) + 1, "untagged": classes.UNTAGGED_COLOUR,
            "kept": review.summary(conn), "refused": text is not None}, status_code=status)

    @app.get("/p/{name}/labeling", response_class=HTMLResponse)
    def labeling_page(request: Request, name: str, note: str = "", n: int = 48):
        """Reads classes.yaml every time, so Reload shows edits made in a text editor."""
        return with_project(request, name, lambda cfg, info, conn: labeling_page_for(
            request, cfg, info, conn, note=note, n_labeled=max(1, min(n, 5000))))

    @app.post("/p/{name}/labeling/classes")
    def labeling_save(request: Request, name: str, text: str = Form("")):
        """Check the edited classes.yaml and save it; on a problem nothing is written and the
        page shows the typed text with each problem's line."""
        def work(cfg, info, conn):
            try:
                classes.save(info["folder"], text)
            except classes.ClassesError as e:
                return labeling_page_for(request, cfg, info, conn, text=text, errors=e.errors,
                                         status=400)
            return RedirectResponse(f"/p/{name}/labeling?" + urlencode(
                {"note": "Saved classes.yaml."}), status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/labeling/classes/form")
    async def labeling_form(request: Request, name: str, check: str = ""):
        """The class form (bite 6, SPEC Q17): JSON {settings, classes}. ?check=1 only checks
        (the form asks after every change); else it saves classes.yaml when every check
        passes. Answers {"errors": [{row, setting, msg}]}, 400 when there are any."""
        try:
            body = await request.json()
        except ValueError:
            body = None

        def work(cfg, info, conn):
            if check:
                errors = classes.form_check(info["folder"], body)[1]
            else:
                errors = classes.form_save(info["folder"], body)
            return JSONResponse({"errors": errors}, 400 if errors else 200)
        return with_project(request, name, work)

    # ---- Label review (S5 bite 3a): frames list, viewer picture, click to select, actions ----

    def review_classes(info) -> list[dict] | None:
        cls = classes.read(info["folder"])
        return cls["data"]["classes"] if cls["data"] else None

    def review_frame_ok(conn, file: str) -> bool:
        row = conn.execute("SELECT status FROM frames WHERE file = ?", (file,)).fetchone()
        return bool(extract.FRAME_NAME.match(file) and row and row[0] in label.LABELED)

    # ---- Export (S5 bite 6): a job that rebuilds export/, then the checks and counts ----

    def export_report(folder: Path) -> dict | None:
        """The last export's summary and checks: export/ (passed) or a newer export.failed/."""
        found = []
        for sub in ("export", "export.failed"):
            p = folder / sub / "summary.json"
            if p.is_file():
                try:
                    rep = json.loads(p.read_text())
                    rep["checks"] = json.loads((folder / sub / "checks.json").read_text())
                except (ValueError, OSError):
                    continue
                found.append({**rep, "folder": str(folder / sub), "failed": sub != "export"})
        return max(found, key=lambda r: r.get("at", "")) if found else None

    @app.get("/p/{name}/export", response_class=HTMLResponse)
    def export_page(request: Request, name: str, error: str = ""):
        def work(cfg, info, conn):
            acc = conn.execute("SELECT COUNT(*) FROM frame_review WHERE accepted = 1").fetchone()[0]
            return templates.TemplateResponse(request, "export.html", {
                "cfg": cfg, "info": info, "accepted": acc, "error": error,
                "job": job_view(jobctl(cfg).latest("export", str(info["folder"]))),
                "report": export_report(info["folder"])})
        return with_project(request, name, work)

    @app.post("/p/{name}/export/start")
    def export_start(request: Request, name: str):
        def work(cfg, info, conn):
            jc = jobctl(cfg)
            if job_view(jc.latest("export", str(info["folder"]))).get("active"):
                return RedirectResponse(f"/p/{name}/export?" + urlencode(
                    {"error": "An export is already queued or running."}), status_code=303)
            if not conn.execute("SELECT 1 FROM frame_review WHERE accepted = 1").fetchone():
                return RedirectResponse(f"/p/{name}/export?" + urlencode(
                    {"error": "No accepted frames yet: accept frames on the Review labels page."}),
                    status_code=303)
            jc.submit("export", {"folder": str(info["folder"])}, project=str(info["folder"]))
            return RedirectResponse(f"/p/{name}/export", status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/labels", response_class=HTMLResponse)
    def labels_page(request: Request, name: str, error: str = ""):
        def work(cfg, info, conn):
            cl = review_classes(info)
            frames = review_labels.frame_list(info["folder"], conn) if cl else []
            return templates.TemplateResponse(request, "review_labels.html", {
                "cfg": cfg, "info": info, "frames": frames, "class_list": cl or [],
                "classes_broken": cl is None, "error": error,
                "missing_types": classes.missing_types(info["folder"], cl) if cl else [],
                "todo": sum(1 for f in frames if not f["accepted"]),
                "totals": review_labels.totals(info["folder"], conn, cl) if cl else [],
                "colours": {c["id"]: classes.colour(c["id"]) for c in cl or []}})
        return with_project(request, name, work)

    @app.post("/p/{name}/labels/add-type")
    def labels_add_type(request: Request, name: str, type: str = Form("")):
        """Append a missing urchin type (e.g. black urchin in a project made before D39) to
        classes.yaml (D50); nothing else in the file changes."""
        def work(cfg, info, conn):
            try:
                classes.add_type(info["folder"], type)
            except classes.ClassesError as e:
                return RedirectResponse(f"/p/{name}/labels?" + urlencode({"error": str(e)}),
                                        status_code=303)
            return RedirectResponse(f"/p/{name}/labels", status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/labels/frame/{file}")
    def labels_frame(request: Request, name: str, file: str):
        def work(cfg, info, conn):
            cl = review_classes(info)
            if not cl or not review_frame_ok(conn, file):
                return JSONResponse({"error": "not found"}, 404)
            try:
                return JSONResponse(review_labels.view(info["folder"], conn, file, cl),
                                    headers={"Cache-Control": "no-store"})
            except (review_state.ReviewError, ValueError) as e:
                return JSONResponse({"error": str(e)}, 404)
        return with_project(request, name, work)

    @app.get("/p/{name}/labels/picture/{file}")
    def labels_picture(request: Request, name: str, file: str, mode: str = "all",
                       sel: int | None = None):
        def work(cfg, info, conn):
            cl = review_classes(info)
            data = (review_labels.picture(info["folder"], conn, file, cl,
                                          mode if mode in review_labels.MODES else "all", sel)
                    if cl and review_frame_ok(conn, file) else None)
            if data is None:
                return PlainTextResponse("not found", status_code=404)
            return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        return with_project(request, name, work)

    @app.get("/p/{name}/labels/closeup/{file}")
    def labels_closeup(request: Request, name: str, file: str, mask: int):
        def work(cfg, info, conn):
            cl = review_classes(info)
            data = (review_labels.closeup(info["folder"], conn, file, cl, mask)
                    if cl and review_frame_ok(conn, file) else None)
            if data is None:
                return PlainTextResponse("not found", status_code=404)
            return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        return with_project(request, name, work)

    @app.post("/p/{name}/labels/add/{file}")
    async def labels_add(request: Request, name: str, file: str):
        """Add a missed object (bite 5b, D46): JSON {point: [x, y]} or {box: [x0, y0, x1, y1]} in
        frame pixels, and class_id (an object class). Queues a "sam_add" job (one GPU job at a
        time, D32) and answers {job_id}; 400 with the reason when the request is not valid."""
        try:
            body = await request.json()
        except ValueError:
            body = None

        def work(cfg, info, conn):
            cl = review_classes(info)
            if not cl or not review_frame_ok(conn, file) or not isinstance(body, dict):
                return JSONResponse({"error": "reload the page"}, 400)
            by_id = {c["id"]: c for c in cl}
            cid = body.get("class_id")
            if not isinstance(cid, int) or cid not in by_id or by_id[cid]["kind"] != "object":
                return JSONResponse({"error": "Pick an object class for the new mask."}, 400)
            point, box = body.get("point"), body.get("box")
            try:
                for v, n in ((point, 2), (box, 4)):
                    if v is not None and not (isinstance(v, list) and len(v) == n
                                              and all(isinstance(x, int) for x in v)):
                        raise ValueError("Bad request: point is [x, y], box is [x0, y0, x1, y1].")
                v = review_labels.view(info["folder"], conn, file, cl)
                add.check_request(v["width"], v["height"], point=point, box=box)
            except (ValueError, review_state.ReviewError) as e:
                return JSONResponse({"error": str(e)}, 400)
            jid = jobctl(cfg).submit("sam_add", label.add_params(info["folder"], file, cid,
                                                                  point=point, box=box),
                                     project=str(info["folder"]))
            return JSONResponse({"job_id": jid})
        return with_project(request, name, work)

    @app.get("/p/{name}/labels/job/{job_id}")
    def labels_job(request: Request, name: str, job_id: int):
        def work(cfg, info, conn):
            job = jobctl(cfg).store.get(job_id)
            if not job or job["kind"] != "sam_add" or job.get("project") != str(info["folder"]):
                return JSONResponse({"error": "not found"}, 404)
            v = job_view(job)
            return JSONResponse({"state": v["state"], "line": v["line"],
                                 "message": job.get("message") or ""},
                                headers={"Cache-Control": "no-store"})
        return with_project(request, name, work)

    @app.get("/p/{name}/labels/at/{file}")
    def labels_at(request: Request, name: str, file: str, x: int, y: int):
        def work(cfg, info, conn):
            cl = review_classes(info)
            if not cl or not review_frame_ok(conn, file):
                return JSONResponse({"error": "not found"}, 404)
            return JSONResponse({"mask_id": review_labels.mask_at(info["folder"], conn, file,
                                                                  cl, x, y)})
        return with_project(request, name, work)

    @app.post("/p/{name}/labels/act/{file}")
    async def labels_act(request: Request, name: str, file: str):
        """One review action, JSON {action, mask_ids?, class_id?}. Answers the frame's new view
        plus a note, or 400 {error} with nothing changed."""
        try:
            body = await request.json()
        except ValueError:
            body = None

        def work(cfg, info, conn):
            cl = review_classes(info)
            if not cl or not review_frame_ok(conn, file) or not isinstance(body, dict):
                return JSONResponse({"error": "reload the page"}, 400)
            try:
                note = review_labels.act(info["folder"], conn, file, cl, body)
            except (review_state.ReviewError, ValueError, TypeError) as e:
                return JSONResponse({"error": str(e)}, 400)
            return JSONResponse({"note": note, "totals": review_labels.totals(info["folder"], conn, cl),
                                 **review_labels.view(info["folder"], conn, file, cl)})
        return with_project(request, name, work)

    def sam_job(cfg, info) -> dict:
        return job_view(jobctl(cfg).latest("sam", str(info["folder"])))

    def sam_busy(cfg, info) -> bool:
        return bool(sam_job(cfg, info).get("active"))

    def start_sam(cfg, info, conn, preview: bool, other: bool = False,
                  estimate: bool = False, start: str = "", stop: str = "",
                  keep_awake: bool = False) -> str:
        """Queue a SAM Preview or Run on the job module; returns a note if it was refused.
        An older paused or interrupted SAM job of this project is cancelled first: the new
        job skips whatever that one finished (D33)."""
        files = label.frames_to_label(conn)
        try:
            classes.ensure(info["folder"])
        except classes.ClassesError:
            return "classes.yaml has problems (listed below); fix them first."
        if not files:
            what = "preview" if preview else "estimate" if estimate else "label"
            return f"No kept frames to {what}. Extract frames or add still images first."
        start_at, stop_by, problem = read_schedule(start, stop)
        if problem:
            return problem
        if "scheduled" in (sam_job(cfg, info).get("state"), extract_job(cfg, info).get("state")):
            return ("A job of this project is scheduled; use its Start now or Cancel first "
                    "(Jobs page).")
        if sam_busy(cfg, info) or extract_busy(cfg, info):
            return ("Another job is running; " + ("preview" if preview else "press Run") +
                    " when it has finished.")
        jc = jobctl(cfg)
        old = sam_job(cfg, info)
        if old.get("state") in ("paused", "interrupted"):
            try:
                jc.cancel(old["id"])
            except JobError:
                pass
        if estimate:   # D35: 1 warm-up + 5 frames spread over the frames Run would label
            cls = classes.ensure(info["folder"])
            work = label.frames_with_work(info["folder"], files,
                                          classes.run_units(cls["classes"]), cls["settings"])
            if not work:
                return "Nothing to estimate: every class is current on every kept frame."
            picks = label.estimate_frames(work)
        else:
            picks = label.preview_frames(files, other=other) if preview else files
        jc.submit("sam", label.job_params(info["folder"], picks, preview=preview,
                                          estimate=estimate),
                  project=str(info["folder"]), start_at=start_at, stop_by=stop_by,
                  keep_awake=keep_awake)
        return ""

    @app.post("/p/{name}/labeling/estimate")
    def labeling_estimate(request: Request, name: str):
        """Time 5 frames after a warm-up frame (D35); their results are kept, so Run skips them."""
        def work(cfg, info, conn):
            note = start_sam(cfg, info, conn, preview=False, estimate=True)
            return RedirectResponse(f"/p/{name}/labeling" + (
                "?" + urlencode({"note": note}) if note else ""), status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/labeling/preview")
    def labeling_preview(request: Request, name: str, other: str = Form("")):
        """Run SAM on 3 kept frames (D26) as a job; one job at a time (D32)."""
        def work(cfg, info, conn):
            note = start_sam(cfg, info, conn, preview=True, other=bool(other))
            return RedirectResponse(f"/p/{name}/labeling" + (
                "?" + urlencode({"note": note}) if note else ""), status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/labeling/run")
    def labeling_run(request: Request, name: str, start: str = Form(""), stop: str = Form(""),
                     keep_awake: str = Form("")):
        """Label every kept frame (D26): units whose results are current are skipped, so Run
        after a pause, a quit or a class change does only what is missing (SPEC Q7). Optional
        Start at / Stop by clock times (D37)."""
        def work(cfg, info, conn):
            note = start_sam(cfg, info, conn, preview=False, start=start, stop=stop,
                             keep_awake=bool(keep_awake))
            return RedirectResponse(f"/p/{name}/labeling" + (
                "?" + urlencode({"note": note}) if note else ""), status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/labeling/status")
    def labeling_status(request: Request, name: str):
        return with_project(request, name, lambda cfg, info, conn: JSONResponse(
            sam_job(cfg, info)))

    LABELED_PAGE = 48   # tiles per "Show more" (bite 6c)

    def labeled_view(info, conn, n: int, since: str = "") -> dict:
        """The "Labeled so far" grid's data: newest first, n tiles, with counts per run unit.
        sig changes whenever a results file is written; when it equals `since` only
        {"sig", "same"} is sent, so polling costs a directory listing."""
        files = label.frames_to_label(conn)
        have = label.labeled_frames(info["folder"], files)
        sig = f"{len(have)}-{max((t for _, t in have), default=0)}-{n}"
        if since == sig:
            return {"sig": sig, "same": True}
        cls = classes.read(info["folder"])
        data = cls["data"]
        units = classes.run_units(data["classes"]) if data else []
        frames = []
        for f, t in have[:n]:
            try:
                c = label.frame_counts(info["folder"], f, units, data["settings"]) if data else {}
            except ValueError:
                continue   # unreadable: Run's plan lists it (QE PR 22)
            frames.append({"file": f, "v": str(t), "caption": f,
                           "counts": {k: v["masks"] for k, v in c.items()},
                           "older": [k for k, v in c.items() if v["older"]]})
        return {"sig": sig, "total": len(have), "kept": len(files), "shown": len(frames),
                "n": n, "frames": frames}

    @app.get("/p/{name}/labeling/labeled")
    def labeling_labeled(request: Request, name: str, n: int = LABELED_PAGE, since: str = ""):
        n = max(1, min(n, 5000))
        return with_project(request, name, lambda cfg, info, conn: JSONResponse(
            labeled_view(info, conn, n, since), headers={"Cache-Control": "no-store"}))

    @app.get("/p/{name}/labeling/overlay/{file}")
    def labeling_overlay(request: Request, name: str, file: str,
                         hide: list[str] = Query(default=[]), map: str = "", size: str = ""):
        """A frame of this project with its saved SAM masks drawn (reads only); one `hide`
        parameter per run unit left out, so a name with a comma works too; size=large for
        the large viewer (bite 5)."""
        def work(cfg, info, conn):
            row = conn.execute("SELECT 1 FROM frames WHERE file = ?", (file,)).fetchone()
            cls = classes.read(info["folder"])
            try:
                data = (label.overlay(info["folder"], file, classes.run_units(
                    cls["data"]["classes"]), set(hide),
                    class_list=cls["data"]["classes"] if map else None,
                    width=label.LARGE_WIDTH if size == "large" else label.PREVIEW_WIDTH)
                    if row and extract.FRAME_NAME.match(file) and cls["data"] else None)
            except ValueError as e:  # unreadable sam_raw file (QE PR 22)
                return PlainTextResponse(str(e), status_code=404)
            except (KeyError, TypeError, AttributeError) as e:  # parses but wrong shape (QE PR 27)
                return PlainTextResponse(f"sam_raw/{file}.json is missing or has a wrong field "
                                         f"({type(e).__name__}: {e}); move it aside and Run "
                                         "again.", status_code=404)
            if data is None:
                return PlainTextResponse("not found", status_code=404)
            return Response(data, media_type="image/jpeg")
        return with_project(request, name, work)

    @app.get("/p/{name}/framethumbs/{file}")
    def frame_thumb(request: Request, name: str, file: str):
        """The grid's scaled-down copy of a frame in the review (by name in the frames table)."""
        def work(cfg, info, conn):
            row = conn.execute("SELECT status FROM frames WHERE file = ?", (file,)).fetchone()
            ok = extract.FRAME_NAME.match(file) and row and row[0] in review.SHOWN_ALL
            path = review.thumbnail(info["folder"], file) if ok else None
            if path is None:
                return PlainTextResponse("not found", status_code=404)
            return FileResponse(path, media_type="image/jpeg")
        return with_project(request, name, work)

    @app.get("/p/{name}/frames/{file}")
    def frame_file(request: Request, name: str, file: str):
        """A kept, filtered-out or rejected frame of this project, by its name in the frames
        table (never a path), from the folder of its status."""
        def work(cfg, info, conn):
            row = conn.execute("SELECT status FROM frames WHERE file = ?", (file,)).fetchone()
            ok = extract.FRAME_NAME.match(file) and row and row[0] in review.SHOWN_ALL
            path = extract.home(info["folder"], row[0]) / file if ok else None
            if not ok or not path.is_file():
                return PlainTextResponse("not found", status_code=404)
            return FileResponse(path, media_type="image/png" if file.endswith(".png")
                                else "image/jpeg")
        return with_project(request, name, work)

    @app.get("/p/{name}/clips/status")
    def clips_status(request: Request, name: str):
        return with_project(request, name, lambda cfg, info, conn: JSONResponse(
            clip_status(cfg, info, ingest.added(conn))))

    def project_file(request, name, code, path_of):
        """Serve a file of one of the project's own videos; codes not in the project get 404."""
        def work(cfg, info, conn):
            codes = {r["path_code"] for r in ingest.added(conn)}
            path = path_of(info["folder"]) if code in codes else None
            if path is None or not path.is_file():
                return PlainTextResponse("not found", status_code=404)
            return FileResponse(path)  # answers Range requests, so the player can seek
        return with_project(request, name, work)

    @app.get("/p/{name}/playback/{code}.mp4")
    def playback(request: Request, name: str, code: str):
        return project_file(request, name, code, lambda f: clips.playback_path(f, code))

    @app.get("/p/{name}/frame/{code}.jpg")
    def frame(request: Request, name: str, code: str, t: float = 0.0):
        """One picture of a video's playback copy at time t: the thumbnail on a clip's row.
        Only at a saved clip's start, so the cache can only grow through the app's own saves
        (QE finding on PR 6: any web page could otherwise ask for thousands of pictures)."""
        def work(cfg, info, conn):
            video = next((r for r in ingest.added(conn) if r["path_code"] == code), None)
            try:
                starts = {round(c["start_s"], 2) for c in clips.load_clips(info["folder"])
                          if video and c["video"] == video["path"]}
            except clips.ClipError:
                starts = set()
            path = (clips.frame_at(info["folder"], code, t, video["duration_s"] or 0)
                    if video and math.isfinite(t) and round(t, 2) in starts else None)
            if path is None:
                return PlainTextResponse("not found", status_code=404)
            return FileResponse(path, media_type="image/jpeg")
        return with_project(request, name, work)

    @app.get("/p/{name}/strip/{code}/{file}")
    def strip(request: Request, name: str, code: str, file: str):
        if not strip_file.match(file):
            return PlainTextResponse("not found", status_code=404)
        return project_file(request, name, code, lambda f: clips.strip_dir(f, code) / file)

    # Build 2 pages live in their own modules (S8: Datasets).
    datasets_pages.register(app, templates, with_project, jobctl, job_view)
    train_pages.register(app, templates, with_project, jobctl, job_view, read_schedule)
    return app


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Same option uvicorn uses: connections of an app copy that just stopped (TIME_WAIT,
        # ~30 s on macOS) must not count as "in use"; a running app still does.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Start the labeling tool in the browser.")
    ap.add_argument("--config", type=Path, default=config.DEFAULT_PATH,
                    help="settings file (default: config.local.toml in this folder)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args(argv)
    url = f"http://{HOST}:{args.port}"
    if not port_free(args.port):
        sys.exit(f"Port {args.port} is already in use. If the app is already running, open "
                 f"{url} in the browser. Otherwise start on another port: "
                 f"uv run app --port {args.port + 1}")
    try:
        app = create_app(args.config)
    except JobError as e:   # another copy already runs on this data folder (QE PR 32)
        sys.exit(str(e))
    print(f"Labeling tool: {url}  (settings: {Path(args.config).absolute()})  Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=HOST, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
