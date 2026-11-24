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

### S0 — Scaffold  `[~]`  (demo pending: Nick)
**Goal:** template filled for this project; full spec in git; data and secrets kept out of git. No app code.
- [x] fill CLAUDE.md, SPEC.md, TRACKER.md, DESIGN.md, DEV_LOG.md, skills, README.md
- [x] commit Nick's spec as `docs/source/mvp_spec.md`
- [x] `.gitignore` for videos, frames, datasets, weights; `config.example.toml` pattern
**Demo (Nick):** run these three from the repo root:
1. `grep -rn "{{[A-Z]*}}" CLAUDE.md README.md docs .claude` → prints nothing.
2. `cp config.example.toml config.local.toml && git status --short` → `config.local.toml` is not listed.
3. `git check-ignore -v test.mp4 frames/x.jpg model.pt` → prints one `.gitignore` rule per file.
**Needs:** nothing.

### S1 — Frames (ingest, clips, extract, review; no SAM)  `[ ]`
**Goal:** spec stages 1–4 work in the browser on Nick's ReefCheck videos.
- [ ] plan nibble: inspect prior art (SPEC §Background), choose what to reuse, confirm stack and start command
- [ ] app starts with one command; dashboard page shows every stage and its status
- [ ] ingest: list videos under `video_roots` with duration, resolution, fps, codec, size; skip sidecar and `._` files
- [ ] clip picker: timeline, thumbnail strip, in/out marks saved to `clips.json`
- [ ] extract frames: sampling, cap, blur filter, near-duplicate filter, source video + timestamp recorded
- [ ] review frames: grid, keyboard keep/delete/undo, delete moves to `rejected/`
**Demo (Nick):** `uv run app` (start command and port confirmed in the S1 plan), open
`http://localhost:8000`. Pick 2 ReefCheck videos, mark a clip with `I`/`O`, extract, delete one
frame with `D`, undo with `Z`, delete it again. The dashboard shows the four stages with their
status and the deleted frame is in the project's `rejected/` folder.
**Needs:** S0; answers to SPEC Q3, Q4, Q6.

### S2 — SAM on single frames  `[ ]`
**Goal:** spec stage 5 without job control: class settings file, prompt preview, tiling, overlap rules.
- [ ] verify SAM facts against primary docs (SPEC Q12) before any code
- [ ] `classes.yaml` loaded and shown in the UI
- [ ] prompt preview on 3 sample frames with masks drawn per class
- [ ] tiling and overlap rules; one result file per frame in `sam_raw/`
**Demo (Nick):** same start command. On the Labeling page press "Preview": 3 frames show coloured
masks per class. Add a class to `classes.yaml`, run again: only the new class is labeled.
**Needs:** S1; Nick has run `hf auth login`; answers to SPEC Q1, Q2, Q12.

### S3 — Job control  `[ ]`
**Goal:** long jobs run in a background worker with estimate, pause/resume, limits and a schedule.
- [ ] estimate (seconds per frame, total, finish time)
- [ ] pause / resume / cancel; progress saved per frame
- [ ] resource limits (device, threads, priority, memory guard)
- [ ] scheduled start, "stop by", keep-awake, log and morning summary
**Demo (Nick):** press "Estimate" → seconds per frame and a finish time. Start, pause, quit the app,
start it again, resume: the frame counter continues with no frame repeated or skipped. Schedule a
start 2 minutes ahead, wait, read the summary page.
**Needs:** S2; answer to SPEC Q8.

### S4 — Review labels and export  `[ ]`
**Goal:** spec stage 6 and all outputs; the spec's acceptance criteria pass on Nick's videos.
- [ ] label review page: class change, delete mask, accept frame, split check, fully-checked flags, running totals
- [ ] export: `detector_coco.json`, `seg_masks/` + `classes.json`, `splits/`, `frame_metrics.csv`
- [ ] export checks shown in the UI (COCO load, mask sizes and ids)
- [ ] README: setup, SAM access steps, 10-minute walkthrough
**Demo (Nick):** review 5 frames, press "Export". The Export page shows the COCO load check PASS,
mask sizes matching their images, and `frame_metrics.csv` values matching those 5 frames. All
acceptance boxes in `docs/source/mvp_spec.md` are ticked.
**Needs:** S3; answers to SPEC Q5, Q9, Q11.

---

## Icebox (captured, not scheduled)

- "Suggest clips" and "Pre-screen with SAM" buttons: sprint not chosen yet (SPEC Q6).
- Later MVP milestones (train, quantize, bench test, quality gate, bake-off, Iridium packet): not this repo's Build 1.
