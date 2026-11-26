# urchin_detector_pipeline

A local labeling tool for the Urchin Monitoring MVP: GoPro videos (or folders of still images) → frames →
SAM 3 labeling → reviewed COCO boxes and outlines for a detector, and per-pixel class masks for a segmenter.
It runs in the browser on one laptop; images never leave it. Build 1 of `docs/source/mvp_spec.md`.

## Setup (once)

Needs macOS on Apple silicon (tested on an M1 Max, 32 GB), `ffmpeg` / `ffprobe` on the PATH, and
[`uv`](https://docs.astral.sh/uv/) (install it with its standalone installer, not Homebrew: Homebrew builds
from source on macOS 14, SPEC facts).

1. Settings: `cp config.example.toml config.local.toml`, then set `video_roots` (folders the app may read
   videos from; never changed) and `data_dir` (where projects, frames, SAM results and exports go; outside the
   repo, e.g. on the SSD). `config.local.toml` is gitignored. The app never creates `data_dir`: plug in the
   disk first.
2. SAM 3 access (Meta's SAM License, text in `THIRD_PARTY_LICENSES`):
   - Make a Hugging Face account and open the model page `facebook/sam3`; if it asks you to request access and
     accept the licence, do so (the SAM 3.1 page does; SPEC facts).
   - Log in once in a terminal: `uvx hf auth login` (the token stays in Hugging Face's own store; the app
     never reads or stores it).
   - Download the weights once into the Hugging Face cache (not into the repo):
     `uvx hf download facebook/sam3 sam3.pt config.json LICENSE`.
3. Start: `uv run app` from the repo folder. The first start installs the Python packages (PyTorch, Meta's SAM
   3 code); then the browser opens the dashboard at `http://127.0.0.1:8000`. The terminal is only for starting
   the app; everything else is in the browser.

## 10-minute walkthrough

The dashboard lists projects; a project page lists the six stages with their status. Long steps run as jobs
(Jobs page: pause, resume, cancel, limits, schedule).

1. **New project** on the dashboard → its page.
2. **Ingest**: pick videos or a folder under `video_roots` (or a folder of still images). GoPro chapters are
   grouped; `.LRV` / `.THM` files are not listed.
3. **Pick clips**: "Make playback copies", then mark in / out with `I` / `O` on the timeline (optional: no clips
   = the whole video).
4. **Extract frames**: one frame every 2 s (cap 300), then the blur and near-duplicate sliders; kept + dropped
   add up to the planned count. Dropped frames are moved, never erased.
5. **Review frames** (optional): `D` deletes (to `rejected/`), `K` keeps, `Z` undoes, `F` full screen.
6. **SAM labeling**: check the classes (form), press **Preview** (3 frames, masks per class), then **Estimate**
   and **Run** (or schedule it overnight with "Keep laptop awake"). Run only labels what changed.
7. **Review labels** (project page → "review"): open a frame; click a mask to select it; type a class or press
   `X` to delete it; urchins come out grey (untagged): `Tab` jumps to the next one and zooms in, `P` / `R` /
   `B` tag it purple / red / black and move on, "Tag all remaining as purple urchin" does the rest; "Add a missed
   object" then click it or drag a box (SAM finds the outline, about 20 s); mark the split check and "fully checked"
   classes; `Enter` accepts the frame (refused while an urchin is untagged); `Z` undoes.
8. **Export** (project page → "export"): press Export. The page shows each check PASS / FAIL, frames per split
   and objects per class. Output in `<data_dir>/projects/<name>/export/`: `images/`, `detector_coco.json`,
   `seg_masks/` + `classes.json`, `splits/`, `frame_metrics.csv`. Re-exporting is always safe.

Fine outline fixes are done outside this tool, in CVAT: create a task with the exported images and upload
`detector_coco.json` as "COCO 1.0" annotations (CVAT's COCO format page lists polygons as supported on import).
Fixes are not imported back in Build 1 (D42).

## Docs

- `docs/source/mvp_spec.md` — Nick's full MVP spec (the source)
- `docs/SPEC.md` — condensed contract: goal, facts, hard constraints, open questions
- `docs/TRACKER.md` — rules + sprint ladder (start here)
- `docs/DESIGN.md`, `docs/DEV_LOG.md`, `docs/PROMPTS.md` — decisions, session log, kickoff prompts

Videos, frames, datasets and model weights never go in git (`.gitignore`). Licences: `THIRD_PARTY_LICENSES`.
Process: created from `template_LLM_project` (agent discipline in `docs/TRACKER.md`; skills in `.claude/skills/`).
