"""The labeling tool's web pages. Start with `uv run app`; it listens on 127.0.0.1 only
(SPEC constraint 1) and opens the dashboard in the default browser.

The settings file is re-read on every page load, so plugging in the SSD or fixing
config.local.toml only needs a browser reload, not a restart.
"""

from __future__ import annotations

import argparse
import socket
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import config, db, project
from app.stages import ingest

HOST = "127.0.0.1"  # never 0.0.0.0: images never leave the laptop (SPEC constraint 1)
# Host and origin rule (DESIGN D14): pages answer only to these host names, and a POST is
# accepted only from a page this app served. Stops other web pages in the browser (and DNS
# rebinding) from reading pages or changing projects.
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
WEB = Path(__file__).parent
templates = Jinja2Templates(directory=WEB / "templates")


def create_app(config_path: Path = config.DEFAULT_PATH) -> FastAPI:
    app = FastAPI(title="Urchin labeling tool", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

    @app.middleware("http")
    async def same_origin_posts_only(request: Request, call_next):
        if request.method not in ("GET", "HEAD"):
            source = request.headers.get("origin") or request.headers.get("referer") or ""
            parts = urlsplit(source)
            if parts.scheme != "http" or parts.netloc != request.headers.get("host"):
                return PlainTextResponse(
                    "Refused: this action can only come from the app's own pages "
                    f"(got origin {source or 'none'!r}).", status_code=403)
        return await call_next(request)

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
    def ingest_page(request: Request, name: str, folder: str = ""):
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
            keys = ingest.added_keys(conn)
            return templates.TemplateResponse(request, "ingest.html", {
                "cfg": cfg, "info": info, "here": here, "error": error,
                "subfolders": subfolders, "videos": videos, "added": ingest.added(conn),
                "added_paths": set(keys.values()),
                # a copy of this video from another folder is already in the project (D13)
                "added_elsewhere": {str(v.path): keys[(v.path.name, v.size)] for v in videos
                                    if (v.path.name, v.size) in keys
                                    and keys[(v.path.name, v.size)] != str(v.path)}})
        return with_project(request, name, work)

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

    @app.post("/p/{name}/ingest/remove")
    def ingest_remove(request: Request, name: str, folder: str = Form(""), path: str = Form("")):
        def work(cfg, info, conn):
            ingest.remove(conn, path)
            return RedirectResponse(ingest_url(name, folder), status_code=303)
        return with_project(request, name, work)

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
