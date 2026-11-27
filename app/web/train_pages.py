"""The Train page (S8 bite 5): pick a dataset version, a model type, classes and a recipe;
Run it as a job (Start at / Stop by, Keep laptop awake); watch the loss curve; the model list.
`register` gets the helpers of create_app it needs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.datasets import loaders, versions
from app.train import job as train_job
from app.train import models, starting


def register(app, templates, with_project, jobctl, job_view, read_schedule) -> None:
    def model_rows(cfg, folder: Path) -> list[dict]:
        """The plain model list: every model with its recipe, epochs done and its job."""
        jobs = {}
        for j in jobctl(cfg).list():
            if j["kind"] == "train":
                jobs.setdefault((j.get("params") or {}).get("model_dir"), j)   # newest first
        rows = []
        for mid in models.list_ids(folder)[::-1]:
            mdir = models.root(folder) / mid
            try:
                rec = models.read_recipe(mdir)
            except models.ModelError as e:
                rows.append({"id": mid, "error": str(e)})
                continue
            got = models.read_metrics(mdir)
            t = rec.get("train", {})
            rows.append({"id": mid, "arch": rec.get("arch", {}), "train": t,
                         "done": len(got), "epochs": t.get("epochs"),
                         "last": got[-1] if got else None,
                         "job": job_view(jobs.get(str(mdir)))})
        return rows

    def start_records(folder: Path) -> dict[str, dict]:
        """model id -> what starting.record needs, for models that finished training."""
        out = {}
        for mid in models.list_ids(folder):
            try:
                t = models.read_recipe(models.root(folder) / mid).get("train", {})
            except models.ModelError:
                continue
            if len(models.read_metrics(models.root(folder) / mid)) >= (t.get("epochs") or 1e9):
                # the sources the run really trained on, as its trainer recorded them: a
                # version that merely holds Urchinbot does not make a model "pre-trained"
                used = [s for r in t.get("recorded", []) for s in r.get("outside_used", [])]
                out[mid] = {"outside": used, "start": t.get("start")}
        return out

    @app.get("/p/{name}/train", response_class=HTMLResponse)
    def train_page(request: Request, name: str, error: str = "", version: int = 0,
                   model: str = ""):
        def work(cfg, info, conn):
            folder = info["folder"]
            found = versions.list_versions(folder)[::-1]
            picked = next((v for v in found if v["number"] == version), found[0] if found else None)
            rows = model_rows(cfg, folder)
            shown = next((r for r in rows if r["id"] == model), rows[0] if rows else None)
            pre = [m for m, r in start_records(folder).items() if r["outside"]]
            return templates.TemplateResponse(request, "train.html", {
                "cfg": cfg, "info": info, "error": error, "versions": found, "picked": picked,
                "classes": loaders.object_classes(picked) if picked else [],
                "default_ids": loaders.default_class_ids(picked) if picked else [],
                "types": train_job.MODEL_TYPES, "models": rows, "shown": shown,
                "pretrained": pre, "not_offered": starting.NOT_OFFERED,
                "job": job_view(jobctl(cfg).latest("train", str(folder)))})
        return with_project(request, name, work)

    @app.post("/p/{name}/train/run")
    async def train_run(request: Request, name: str, version: int = Form(0),
                        model_type: str = Form(""), start_kind: str = Form("scratch"),
                        start_model: str = Form(""), start: str = Form(""),
                        stop: str = Form(""), keep_awake: str = Form("")):
        form = await request.form()

        def work(cfg, info, conn):
            folder, jc = info["folder"], jobctl(cfg)

            def back(problem: str):
                return RedirectResponse(f"/p/{name}/train?" + urlencode(
                    {"error": problem, "version": version}), status_code=303)
            if job_view(jc.latest("train", str(folder))).get("state") in (
                    "scheduled", "queued", "running", "pausing", "paused", "interrupted"):
                return back("A training job of this project is waiting or running: finish, "
                            "resume or cancel it first.")
            start_at, stop_by, problem = read_schedule(start, stop)
            if problem:
                return back(problem)
            try:
                manifest, _ = loaders.load(folder, version)
                ids = loaders.check_classes(manifest, form.getlist("classes"))
                choice = {"kind": start_kind, "model": start_model}
                began = starting.record(choice, start_records(folder))
                key, recipe = train_job.build_recipe(model_type, dict(form), manifest, ids, began)
                seen, _ = train_job.usable(
                    loaders.detector_samples(folder, version, "train", ids)["samples"], ids)
                if not any(s["boxes"] for s in seen):
                    raise ValueError(
                        f"Version {version} gives training no object of these classes: a frame "
                        "is used only when every picked class is \"fully checked\" on it. On the "
                        "Review page mark them \"fully checked\" on the frames you reviewed, "
                        "export and freeze again (the Datasets page shows what is skipped).")
                mdir = models.create(folder, key, recipe)
            except (loaders.DatasetError, starting.StartError, models.ModelError,
                    ValueError) as e:
                return back(str(e))
            jc.submit("train", {"model_dir": str(mdir), "trainer": recipe["train"]["trainer"]},
                      project=str(folder), start_at=start_at, stop_by=stop_by,
                      keep_awake=bool(keep_awake))
            return RedirectResponse(f"/p/{name}/train?" + urlencode({"model": mdir.name}),
                                    status_code=303)
        return with_project(request, name, work)

    @app.get("/p/{name}/train/metrics/{model_id}")
    def train_metrics(request: Request, name: str, model_id: str):
        """A model's finished epochs (train.jsonl) for the curves; read only."""
        def work(cfg, info, conn):
            try:
                mdir = models.model_dir(info["folder"], model_id)
                epochs = models.read_recipe(mdir)["train"]["epochs"]
            except (models.ModelError, KeyError) as e:
                return JSONResponse({"error": str(e)}, status_code=404)
            return JSONResponse({"model": model_id, "epochs": epochs,
                                 "rows": models.read_metrics(mdir)},
                                headers={"Cache-Control": "no-store"})
        return with_project(request, name, work)
