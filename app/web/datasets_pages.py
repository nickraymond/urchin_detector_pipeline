"""The Datasets page (S8 bite 2a): freeze a dataset version from the export, list the versions.
Kept out of server.py so Build 2's pages do not all land in one file; `register` gets the few
helpers of create_app it needs."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.datasets import loaders, versions, view
from app.export.splits import SPLITS


def register(app, templates, with_project, jobctl, job_view) -> None:
    def page_url(name: str, **query) -> str:
        return f"/p/{name}/datasets" + ("?" + urlencode(query) if query else "")

    @app.get("/p/{name}/datasets", response_class=HTMLResponse)
    def datasets_page(request: Request, name: str, error: str = "",
                      classes: list[int] = Query(default=[])):
        def work(cfg, info, conn):
            try:
                export, why_not = versions.export_ready(info["folder"]), ""
            except versions.FreezeError as e:
                export, why_not = None, str(e)
            found = versions.list_versions(info["folder"])
            for v in found:     # bite 2b: what a detector run would get from this version
                try:
                    v["view"] = view.training_view(info["folder"], v["number"], classes or None)
                except loaders.DatasetError as e:
                    v["view"], v["view_error"] = None, str(e)
            return templates.TemplateResponse(request, "datasets.html", {
                "cfg": cfg, "info": info, "error": error, "export": export, "why_not": why_not,
                "versions": found[::-1], "splits": SPLITS, "name_max": versions.NAME_MAX,
                "job": job_view(jobctl(cfg).latest("freeze", str(info["folder"])))})
        return with_project(request, name, work)

    @app.post("/p/{name}/datasets/freeze")
    def datasets_freeze(request: Request, name: str, version_name: str = Form("")):
        def work(cfg, info, conn):
            jc, folder = jobctl(cfg), str(info["folder"])
            problem = ""
            if job_view(jc.latest("freeze", folder)).get("active"):
                problem = "A freeze is already queued or running."
            elif job_view(jc.latest("export", folder)).get("active"):
                problem = "An export is queued or running: wait for it, then freeze."
            elif len(" ".join(version_name.split())) > versions.NAME_MAX:
                problem = f"The name is too long (at most {versions.NAME_MAX} characters)."
            else:
                try:
                    versions.export_ready(info["folder"])
                except versions.FreezeError as e:
                    problem = str(e)
            if problem:
                return RedirectResponse(page_url(name, error=problem), status_code=303)
            jc.submit("freeze", {"folder": folder, "name": version_name}, project=folder)
            return RedirectResponse(page_url(name), status_code=303)
        return with_project(request, name, work)
