"""The Train page (S8 bite 5): pick a dataset version, a model type, classes and a recipe;
Estimate (time a few steps) or Run it as a job (Start at / Stop by, Keep laptop awake); watch
the loss curves and a few validation frames with boxes drawn; the model list.
`register` gets the helpers of create_app it needs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

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
                rec = models.read_recipe(models.root(folder) / mid)
                t = rec.get("train", {})
            except models.ModelError:
                continue
            if len(models.read_metrics(models.root(folder) / mid)) >= (t.get("epochs") or 1e9):
                # the sources the run really trained on, as its trainer recorded them: a
                # version that merely holds Urchinbot does not make a model "pre-trained"
                # (once each: a paused run has one record per start)
                used = list({s["key"]: s for r in t.get("recorded", [])
                             for s in r.get("outside_used", [])}.values())
                out[mid] = {"outside": used, "start": t.get("start"), "arch": rec.get("arch", {})}
        return out

    def plan(folder: Path, version: int, form, model_type: str, start_kind: str,
             start_model: str, cfg_outside=lambda: None) -> tuple[str, dict]:
        """(arch key, recipe) for the form as sent, for Run and for Estimate alike. Raises
        with a message for the page when the run cannot be made."""
        manifest, _ = loaders.load(folder, version)
        on_outside = form.get("data") == "outside"      # pre-training: one class, no class pick
        ids = [] if on_outside else loaders.check_classes(manifest, form.getlist("classes"))
        records = start_records(folder)
        began = starting.record({"kind": start_kind, "model": start_model}, records)
        key, recipe = train_job.build_recipe(model_type, dict(form), manifest, ids, began)
        if began["kind"] == "pretrained":    # the weights must fit: same family, size, first layer
            a, b = recipe["arch"], records[start_model]["arch"]
            if any(a.get(k) != b.get(k) for k in ("family", "size", "variant")):
                raise ValueError(f"Model {start_model} is a different model type or size: a run "
                                 "can only start from a pre-trained model of its own type.")
        if on_outside:
            if not cfg_outside():
                raise ValueError("outside_data_dir is not set in config.local.toml (or the "
                                 "folder is missing): a run on outside data needs it.")
            return key, recipe
        full, boxes_only = loaders.run_frames(
            loaders.detector_samples(folder, version, "train", ids)["samples"], ids)
        loaders.need_validation(loaders.usable(
            loaders.detector_samples(folder, version, "valid", ids)["samples"], ids)[0])
        if not any(s["boxes"] for s in full + boxes_only):
            raise ValueError(
                f"Version {version} gives training no object of these classes: a class "
                "counts on a frame only where it is \"fully checked\". On the "
                "Review page mark them \"fully checked\" on the frames you reviewed, "
                "export and freeze again (the Datasets page shows what is skipped).")
        return key, recipe

    def frame_counts(folder: Path, picked, class_ids) -> tuple[dict | None, str]:
        """Frames a run of these classes would get on this version (D66): in full, boxes only, validation.
        (None, "") with no class ticked; (None, why) when the version cannot be read."""
        if not picked or not class_ids:
            return None, ""
        try:
            kinds = {s: loaders.run_frames(loaders.detector_samples(
                folder, picked["number"], s, class_ids)["samples"], class_ids) for s in ("train", "valid")}
        except loaders.DatasetError as e:
            return None, str(e)
        return {"full": len(kinds["train"][0]), "boxes_only": len(kinds["train"][1]),
                "valid": len(kinds["valid"][0])}, ""

    @app.get("/p/{name}/train/counts", response_class=HTMLResponse)
    def train_counts(request: Request, name: str, version: int = 0, classes: list[str] = Query(default=[])):
        """The frame counts for the classes ticked now (QE PR 93): the same span the page renders."""
        def work(cfg, info, conn):
            found = versions.list_versions(info["folder"])
            picked = next((v for v in found if v["number"] == version), None)
            if picked is None:
                return HTMLResponse(f"No dataset version {version}.", status_code=404)
            ids, error = [], ""
            try:
                ids = loaders.check_classes(loaders.load(info["folder"], version)[0], classes) if classes else []
            except (ValueError, loaders.DatasetError) as e:
                error = str(e)
            counts, counts_error = frame_counts(info["folder"], picked, ids) if not error else (None, error)
            return templates.TemplateResponse(request, "_frame_counts.html", {
                "info": info, "picked": picked, "frame_counts": counts, "counts_error": counts_error})
        return with_project(request, name, work)

    def outside_param(cfg, recipe: dict) -> dict:
        """The outside data folder goes into the job file only for a run that reads it."""
        return ({"outside_dir": str(cfg.outside_data_dir)}
                if recipe["train"].get("data") == "outside" else {})

    BUSY = ("scheduled", "queued", "running", "pausing", "paused", "interrupted")

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
            est = train_job.read_estimate(folder)
            # the form starts from the last Estimate's settings (so Run runs what was timed)
            # when it was made on the version shown, else from the first model type's defaults
            last = (est or {}).get("recipe", {}).get("train", {})
            same = bool(picked) and last.get("dataset", {}).get("version") == picked["number"] \
                and last.get("model_type") in train_job.MODEL_TYPES
            tkey = last["model_type"] if same else next(iter(train_job.MODEL_TYPES), "")
            d = train_job.MODEL_TYPES.get(tkey, {}).get("defaults", {})
            form = {"model_type": tkey, "data": last.get("data", "project") if same else "project",
                    "input": last["tile"] if same else d.get("input"),
                    "epochs": last["epochs"] if same else d.get("epochs"),
                    "augment": last["augment"] if same else d.get("augment"),
                    "seed": last["seed"] if same else d.get("seed"),
                    "val_every": last.get("val_every", d.get("val_every")) if same else d.get("val_every"),
                    "class_ids": last["class_ids"] if same and last["class_ids"]
                    else loaders.default_class_ids(picked) if picked else []}
            counts, counts_error = frame_counts(folder, picked, form["class_ids"])
            no_background = ""      # for the model shown: was it ever taught "no object"? (D66)
            if shown and not shown.get("error") and shown["train"].get("data") != "outside":
                try:
                    ids = shown["train"]["class_ids"]
                    no_background = loaders.background_warning(*loaders.run_frames(
                        loaders.detector_samples(folder, shown["train"]["dataset"]["version"],
                                                 "train", ids)["samples"], ids))
                except (loaders.DatasetError, KeyError):
                    no_background = ""
            return templates.TemplateResponse(request, "train.html", {
                "frame_counts": counts, "counts_error": counts_error, "no_background": no_background,
                "cfg": cfg, "info": info, "error": error, "versions": found, "picked": picked,
                "classes": loaders.object_classes(picked) if picked else [],
                "form": form, "estimate": est, "estimate_steps": train_job.ESTIMATE_STEPS,
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
            if job_view(jc.latest("train", str(folder))).get("state") in BUSY:
                return back("A training job of this project is waiting or running: finish, "
                            "resume or cancel it first.")
            start_at, stop_by, problem = read_schedule(start, stop)
            if problem:
                return back(problem)
            try:
                key, recipe = plan(folder, version, form, model_type, start_kind, start_model,
                                   lambda: cfg.outside_data_dir)
                mdir = models.create(folder, key, recipe)
            except (loaders.DatasetError, starting.StartError, models.ModelError,
                    ValueError) as e:
                return back(str(e))
            jc.submit("train", {"model_dir": str(mdir), "trainer": recipe["train"]["trainer"],
                                **outside_param(cfg, recipe)},
                      project=str(folder), start_at=start_at, stop_by=stop_by,
                      keep_awake=bool(keep_awake))
            return RedirectResponse(f"/p/{name}/train?" + urlencode({"model": mdir.name}),
                                    status_code=303)
        return with_project(request, name, work)

    @app.post("/p/{name}/train/estimate")
    async def train_estimate(request: Request, name: str, version: int = Form(0),
                             model_type: str = Form(""), start_kind: str = Form("scratch"),
                             start_model: str = Form("")):
        """Estimate (spec stage 8; the D35 pattern): a short job that times a few steps of the
        run the form describes. It makes no model; its recipe and result live in
        models/.estimate/ and replace the estimate before."""
        form = await request.form()

        def work(cfg, info, conn):
            folder, jc = info["folder"], jobctl(cfg)

            def back(problem: str):
                return RedirectResponse(f"/p/{name}/train?" + urlencode(
                    {"error": problem, "version": version}), status_code=303)
            if job_view(jc.latest("train", str(folder))).get("state") in BUSY:
                return back("A training job of this project is waiting or running: finish, "
                            "resume or cancel it first.")
            try:
                _, recipe = plan(folder, version, form, model_type, start_kind, start_model,
                                 lambda: cfg.outside_data_dir)
            except (loaders.DatasetError, starting.StartError, models.ModelError,
                    ValueError) as e:
                return back(str(e))
            edir = train_job.estimate_dir(folder)
            edir.mkdir(parents=True, exist_ok=True)
            (edir / "estimate.json").unlink(missing_ok=True)   # never show the old one as new
            models.write_json(edir / "recipe.json", recipe)
            jc.submit("train", {"model_dir": str(edir), "trainer": recipe["train"]["trainer"],
                                "estimate": True, **outside_param(cfg, recipe)},
                      project=str(folder))
            return RedirectResponse(f"/p/{name}/train?" + urlencode({"version": version}),
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
            vdir = mdir / "val_frames"
            frames = sorted(int(p.stem[4:]) for p in vdir.glob("val_*.png")
                            if p.stem[4:].isdigit()) if vdir.is_dir() else []
            try:
                val_epoch = int((vdir / "epoch.txt").read_text())
            except (OSError, ValueError):
                val_epoch = None
            return JSONResponse({"model": model_id, "epochs": epochs,
                                 "rows": models.read_metrics(mdir),
                                 "val_frames": frames, "val_epoch": val_epoch},
                                headers={"Cache-Control": "no-store"})
        return with_project(request, name, work)

    @app.get("/p/{name}/train/val/{model_id}/{index}")
    def train_val_frame(request: Request, name: str, model_id: str, index: int):
        """One validation canvas with boxes drawn (the trainer rewrites it each epoch)."""
        def work(cfg, info, conn):
            try:
                p = models.model_dir(info["folder"], model_id) / "val_frames" / f"val_{index}.png"
            except models.ModelError as e:
                return JSONResponse({"error": str(e)}, status_code=404)
            if not p.is_file():
                return JSONResponse({"error": "no such validation frame"}, status_code=404)
            return FileResponse(p, media_type="image/png",
                                headers={"Cache-Control": "no-store"})
        return with_project(request, name, work)
