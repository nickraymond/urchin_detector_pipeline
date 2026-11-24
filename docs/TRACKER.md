# TRACKER.md — Sprint Ladder & Rules

*The agent entry point. Newest state lives here.*
*Last updated: 2026-11-24 · Owner/gate: **Nick***

---

## Rules for Agents (READ FIRST, EVERY SESSION)

1. **Read this whole document cover-to-cover first, every session.** Then skim
   `docs/SPEC.md` and `docs/DESIGN.md`. Read the top ~3 entries of
   `docs/DEV_LOG.md`.
2. **Take small code bites.** One TODO at a time, target ~300 LoC. If SPEC.md
   is too thin to inform the bite, stop and ask Nick.
3. **Four nibbles per bite:**
   1. **Plan** — throwaway code ok; change no files. *Gate: Nick approves.*
   2. **Code + unit tests** — flag Nick on substantial plan changes.
   3. **Manual tests** — Nick runs them; provide copy-pastable CLI.
   4. **Open PR.**
4. **Feature branch for all new work** — `sprint/<n>-<name>`, PR to `main`. Never
   commit to main.
5. **Every sprint ends with a live demo Nick can run** — exact commands in
   the sprint's Demo section and the PR description.
6. **End of every session:** DEV_LOG.md entry (newest on top); DESIGN.md updated
   on any architecture/decision change.
7. **Facts carry sources; unknowns get flagged, not guessed.**
8. **Data and secrets: SPEC §Hard constraints is absolute.** Stop and report before
   anything would send an image off the laptop, write to a source video, put data,
   weights or a token in git, or add a model or library whose license is not verified.

### Project layout

```
docs/     SPEC.md TRACKER.md DESIGN.md DEV_LOG.md PROMPTS.md  source/mvp_spec.md
app/      the labeling tool: web UI, worker, one module per stage (from S1)
tests/    unit tests + small synthetic fixtures (no real frames in git) (from S1)
config.example.toml   copy to config.local.toml (gitignored) and set your paths
```

---

## Sprint ladder

State key: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked

### S0 — Scaffold  `[x]`  (demo passed: Nick 2026-11-24; PR 1 squash-merged to main as `8c322ac`)
**Goal:** template filled for this project; full spec in git; data and secrets kept out of git. No app code.
- [x] fill CLAUDE.md, SPEC.md, TRACKER.md, DESIGN.md, DEV_LOG.md, skills, README.md
- [x] commit Nick's spec as `docs/source/mvp_spec.md`
- [x] `.gitignore` for videos, frames, datasets, weights; `config.example.toml` pattern
- [x] spec edits from Nick's review (2026-11-24) and QE fixes
**Demo (Nick):** run these three from the repo root:
1. `grep -rn "{{[A-Z]*}}" CLAUDE.md README.md docs .claude` → prints nothing.
2. `cp config.example.toml config.local.toml && git status --short` → `config.local.toml` is not listed.
3. `git check-ignore -v test.mp4 frames/x.jpg model.pt` → prints one `.gitignore` rule per file.
**Needs:** nothing.

### S1 — SAM test on the laptop  `[~]`
**Goal:** know whether SAM 3.1 runs on Nick's M1 Max, how fast, and under what licence,
before any tool code is written. This is a test, not the tool: keep it small and throwaway-friendly.
- [~] re-read the primary docs and record what they say in SPEC (Q12): `facebookresearch/sam3`
      readme and its SAM 3.1 release notes; the `LICENSE` file in `facebook/sam3.1`
- [x] install the official SAM 3 code in a local `.venv` with `uv` (never Homebrew; SPEC facts)
- [ ] download the weights with `uvx hf download` into the Hugging Face cache (not the repo, not git)
- [x] pull 3 frames from one ReefCheck video with ffmpeg into `<data_dir>/sam_test/`
- [ ] run text prompts ("sea urchin", "purple sea urchin", "red sea urchin", "sea star",
      "sunflower sea star", "sea anemone", "fish") on each
      frame: first on the Apple GPU (MPS), then CPU if MPS fails; at full size with tiles and at
      one reduced size
- [ ] on one frame, also test a point prompt (one click on an urchin) and a box prompt, as
      the "add a missed object" step in label review needs them
- [ ] record per run: device, seconds per frame (first frame not counted), peak memory,
      masks found; and any error text in full
- [x] write one local report page, `<data_dir>/sam_test/report.html`: each frame with its masks
      drawn, and a table of the numbers. No internet assets.
- [ ] if SAM 3.1 will not run: try `facebook/sam3` the same way, then stop and report the options
      (SAM 3, CPU only, another machine). Time box: one working day.
**Demo (Nick):** `uv run sam-test --video "<one ReefCheck video>"` runs the test and opens
`report.html` in the browser: 3 frames with coloured masks on the urchins, and a table showing
device, seconds per frame and memory. A failed run still produces the page, with the error shown.
**Needs:** S0; Nick logged in to Hugging Face (done 2026-11-24); the SSD plugged in.

### S2 — Frames (ingest, clips, extract, review; no SAM)  `[ ]`
**Goal:** spec stages 1–4 work in the browser on Nick's ReefCheck videos.
- [ ] plan nibble: inspect prior art (SPEC §Background), choose what to reuse, confirm stack and start command
- [ ] app starts with one command; dashboard page shows every stage and its status
- [ ] ingest: list videos under `video_roots` with duration, resolution, fps, codec, size; skip
      sidecar and `._` files; also accept folders of still images
- [ ] HEVC check: warn if the decoder is missing; if the browser cannot play a video, scrub a
      low-res proxy (GoPro `.LRV` or an H.264 proxy made with ffmpeg)
- [ ] clip picker: timeline, thumbnail strip, in/out marks saved to `clips.json`
- [ ] extract frames: sampling, cap, blur filter, near-duplicate filter, source video + timestamp recorded
- [ ] review frames: grid, keyboard keep/delete/undo, delete moves to `rejected/`
- [ ] each stage resumes where it left off and can be re-run without redoing earlier stages
**Demo (Nick):** `uv run app` (start command and port confirmed in the S2 plan), open
`http://localhost:8000`. Pick 2 ReefCheck videos, mark a clip with `I`/`O`, extract: the kept and
dropped counts add up to the preview count. Delete one frame with `D`, undo with `Z`, delete it
again. Quit and restart: the dashboard shows the four stages where you left them, and the deleted
frame is in the project's `rejected/` folder.
**Needs:** S1 result (so the stack fits what SAM needs); answer to SPEC Q6.

### S3 — SAM on single frames  `[ ]`
**Goal:** spec stage 5 without job control: class settings file, prompt preview, tiling, overlap rules.
- [ ] `classes.yaml` loaded and shown in the UI
- [ ] prompt preview on 3 sample frames with masks drawn per class
- [ ] tiling and overlap rules; one result file per frame in `sam_raw/`, recording prompt and settings
- [ ] re-run only what changed; reviewed masks kept and marked "from older settings"
**Demo (Nick):** same start command. On the Labeling page press "Preview": 3 frames show coloured
masks per class. Add a class to `classes.yaml`, run again: only the new class is labeled.
**Needs:** S2; answers to SPEC Q2, Q10, Q14.

### S4 — Job control  `[ ]`
**Goal:** long jobs run in a background worker with estimate, pause/resume, limits and a schedule.
Built as its own module with no labeling-specific code (DESIGN D4).
- [ ] estimate (seconds per frame, total, finish time)
- [ ] pause / resume / cancel; progress saved per frame
- [ ] resource limits (device, threads, priority, memory guard)
- [ ] scheduled start, "stop by", "Keep laptop awake" toggle with reminder, log and morning summary
**Demo (Nick):** press "Estimate" → seconds per frame and a finish time. Start a run and keep
using the laptop (browser, a video): it stays usable with the thread cap on. Pause, quit the app,
start it again, resume: the frame counter continues with no frame repeated or skipped. Schedule a
start 2 minutes ahead with "Keep laptop awake" on, wait, read the summary page.
**Needs:** S3.

### S5 — Review labels and export  `[ ]`
**Goal:** spec stage 6 and all outputs; the spec's acceptance criteria pass on Nick's videos.
- [ ] label review page: class change, delete mask, add a missed object by click or box, accept
      frame, split check, fully-checked flags, running totals
- [ ] export: `export/images/`, `detector_coco.json`, `seg_masks/` + `classes.json`, `splits/`
      (by site and dive date), `frame_metrics.csv`
- [ ] export checks shown in the UI (COCO load, mask sizes and ids)
- [ ] `THIRD_PARTY_LICENSES` file with the SAM License text
- [ ] README: setup, SAM access steps, 10-minute walkthrough
**Demo (Nick):** review 5 frames, add one missed urchin by clicking it, press "Export". The Export
page shows the COCO load check PASS, mask sizes matching their images, and `frame_metrics.csv`
values matching those 5 frames. All acceptance boxes in `docs/source/mvp_spec.md` are ticked.
**Needs:** S4; answers to SPEC Q9, Q11.

---

## Icebox (captured, not scheduled)

- "Suggest clips" and "Pre-screen with SAM" buttons: sprint not chosen yet (SPEC Q6).
- Kelp and kelp shoot classes: when Nick has reference images (SPEC Q1, answered).
- Later MVP milestones (train, quantize, bench test, quality gate, bake-off, Iridium packet): not this repo's Build 1.
