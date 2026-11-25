"""The labeling tool's web pages. Start with `uv run app`; it listens on 127.0.0.1 only
(SPEC constraint 1) and opens the dashboard in the default browser.

The settings file is re-read on every page load, so plugging in the SSD or fixing
config.local.toml only needs a browser reload, not a restart.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import socket
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import config, db, project
from app.jobs import Jobs
from app.stages import clips, extract, ingest, review

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

    def dashboard(request: Request, error: str = "", name: str = "", status: int = 200):
        cfg = config.load(config_path)
        projects = project.list_projects(cfg.data_dir) if cfg.ok else []
        return templates.TemplateResponse(request, "dashboard.html", {
            "cfg": cfg, "projects": projects, "error": error, "name": name,
            "name_help": project.NAME_HELP}, status_code=status)

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

    @app.get("/p/{name}", response_class=HTMLResponse)
    def project_page(request: Request, name: str):
        return with_project(request, name, lambda cfg, info, conn: templates.TemplateResponse(
            request, "project.html", {"cfg": cfg, "info": info,
                                      "stages": db.stage_statuses(conn)}))

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
                    note = (f"Added {n['added']} still images" +
                            (f"; {n['skipped']} were already in the project" if n["skipped"] else "") +
                            (f"; {n['unreadable']} could not be read and were left out"
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

    jobs = Jobs()
    app.state.jobs = jobs
    strip_file = re.compile(r"^\d{4}\.jpg$")

    def clip_status(info, videos) -> dict:
        out = {}
        for v in videos:
            code = v["path_code"]
            st = jobs.status(f"{info['folder']}:{code}") or {}
            out[code] = {**st, "ready": clips.ready(info["folder"], code)}
        return out

    @app.get("/p/{name}/clips", response_class=HTMLResponse)
    def clips_page(request: Request, name: str, v: str = ""):
        def work(cfg, info, conn):
            videos = ingest.added(conn)
            current = next((r for r in videos if r["path_code"] == v), videos[0] if videos else None)
            status = clip_status(info, videos)
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
            for row in ingest.added(conn):
                if not clips.ready(info["folder"], row["path_code"]):
                    jobs.submit(f"{info['folder']}:{row['path_code']}",
                                lambda report, row=row: clips.make_playback(
                                    row, info["folder"], report))
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

    def extract_busy(info) -> bool:
        return (jobs.status(f"{info['folder']}:extract") or {}).get("state") in ("queued",
                                                                                 "running")

    @app.get("/p/{name}/extract", response_class=HTMLResponse)
    def extract_page(request: Request, name: str, error: str = ""):
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
                "job": jobs.status(f"{info['folder']}:extract") or {}, "frames": frames,
                "blur": blur, "dup": dup, "error": error,
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
            if not problem and extract_busy(info):
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
                      cap: int = Form(300)):
        def work(cfg, info, conn):
            problem = check_settings(every, cap)
            if problem:
                return RedirectResponse(f"/p/{name}/extract?" + urlencode({"error": problem}),
                                        status_code=303)
            project.update_settings(info["folder"], seconds_between_samples=every, frame_cap=cap)
            samples, _ = extract_plan(info, conn, every, cap)
            blur, dup = filter_settings(info)
            jobs.submit(f"{info['folder']}:extract",
                        lambda report: extract.run(info["folder"], samples, report, blur, dup))
            return RedirectResponse(f"/p/{name}/extract", status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/extract/status")
    def extract_status(request: Request, name: str):
        return with_project(request, name, lambda cfg, info, conn: JSONResponse({
            "job": jobs.status(f"{info['folder']}:extract") or {},
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
                "busy": extract_busy(info)})
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
            if extract_busy(info):
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
            clip_status(info, ingest.added(conn))))

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
    print(f"Labeling tool: {url}  (settings: {Path(args.config).absolute()})  Ctrl-C to stop.")
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(create_app(args.config), host=HOST, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
