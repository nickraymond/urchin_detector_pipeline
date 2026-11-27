"""Starting weights: a recorded choice per training run (S8 bite 3; spec stage 7).

Three kinds, and what each needs before a run may use it:

- "scratch": nothing.
- "pretrained": a model trained in this app on outside data. Every source it was trained on
  must be an allowed outside source (app/datasets/outside.py) and it must itself have started
  from scratch or from another such model, so no licence is hidden down the chain.
- "published": weights someone else released. Offered only when listed in PUBLISHED with a
  recorded permissive licence. None is listed today: YOLOX's COCO weights carry no licence
  statement of their own (THIRD_PARTY_LICENSES), and Nick's earlier YOLOX weights were trained
  on the DUO set (Nick, 2026-11-27: not a starting point for a camera model).

`record()` returns what the run stores in its run record, so a model always says what it
started from and under which licences.
"""

from __future__ import annotations

from app.datasets import outside

KINDS = ("scratch", "pretrained", "published")
PUBLISHED: dict[str, dict] = {}     # name -> {"licence", "licence_source", "file"}
NOT_OFFERED = [
    {"name": "YOLOX COCO weights (Megvii release files)",
     "why": "No licence stated for the weights; COCO's images stay under Flickr's terms."},
    {"name": "Earlier YOLOX stage-1 weights (~/nereus_ml/runs)",
     "why": "Trained on Urchinbot + DUO + RF100; no DUO-trained weights (Nick, 2026-11-27)."},
]


class StartError(ValueError):
    """The starting weights asked for cannot be used; the message says why."""


def record(choice: dict | None, models: dict[str, dict] | None = None) -> dict:
    """Check a choice ({"kind": ..., "model": id | "name": ...}) and return the record for
    the run. `models` maps a model id to its run record ({"start": <record>, "outside":
    [{"key", "licence"}], ...}); only needed for "pretrained"."""
    choice = choice or {"kind": "scratch"}
    kind = choice.get("kind")
    if kind == "scratch":
        return {"kind": "scratch", "licences": []}
    if kind == "published":
        pub = PUBLISHED.get(choice.get("name") or "")
        if pub is None:
            raise StartError("No published weights are offered: none has a recorded permissive "
                             "licence (THIRD_PARTY_LICENSES). Start from scratch or from a "
                             "model pre-trained here on the allowed outside data.")
        return {"kind": "published", "name": choice["name"], "licences": [pub["licence"]],
                "licence_source": pub["licence_source"], "file": pub["file"]}
    if kind == "pretrained":
        run = (models or {}).get(choice.get("model") or "")
        if run is None:
            raise StartError(f"No model {choice.get('model')!r} to start from.")
        sources = run.get("outside") or []
        if not sources:
            raise StartError(f"Model {choice['model']} was not trained on outside data, so it "
                             "is not a pre-trained start. Pick one that was, or scratch.")
        bad = [s["key"] for s in sources
               if not outside.BY_KEY.get(s["key"], {}).get("allowed")]
        if bad:
            raise StartError(f"Model {choice['model']} was trained on {', '.join(bad)}, which "
                             "is not an allowed outside source.")
        parent = run.get("start") or {"kind": "scratch"}
        if parent.get("kind") not in ("scratch", "pretrained"):
            raise StartError(f"Model {choice['model']} itself started from {parent.get('kind')} "
                             "weights: not a clean pre-trained start.")
        licences = sorted({outside.BY_KEY[s["key"]]["licence"] for s in sources}
                          | set(parent.get("licences") or []))
        return {"kind": "pretrained", "model": choice["model"], "licences": licences,
                "sources": [{"key": s["key"], "licence": outside.BY_KEY[s["key"]]["licence"]}
                            for s in sources]}
    raise StartError(f"Unknown starting weights {kind!r}. Pick one of: {', '.join(KINDS)}.")
