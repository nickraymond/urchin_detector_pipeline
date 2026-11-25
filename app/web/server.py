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
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import config, db, project

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

    @app.get("/p/{name}", response_class=HTMLResponse)
    def project_page(request: Request, name: str):
        cfg = config.load(config_path)
        if not cfg.ok:
            return dashboard(request, status=503)
        try:
            info = project.open_project(cfg.data_dir, name)
            conn = db.connect(info["folder"] / "state.sqlite")
            try:
                stages = db.stage_statuses(conn)
            finally:
                conn.close()
        except (project.ProjectError, OSError, sqlite3.Error) as e:
            return dashboard(request, error=str(e), status=404)
        return templates.TemplateResponse(request, "project.html", {
            "cfg": cfg, "info": info, "stages": stages})

    return app


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
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
