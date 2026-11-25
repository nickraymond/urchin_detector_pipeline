# TRACKER.md — Sprint Ladder & Rules

*The agent entry point. Newest state lives here.*
*Last updated: 2026-11-25 · Owner/gate: **Nick***

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

### S1 — SAM test on the laptop  `[x]`  (demo passed: Nick 2026-11-24; PR 2 squash-merged to main as `a362495`)
**Goal:** know whether SAM 3.1 runs on Nick's M1 Max, how fast, and under what licence,
before any tool code is written. This is a test, not the tool: keep it small and throwaway-friendly.
- [x] re-read the primary docs and record what they say in SPEC (Q12): `facebookresearch/sam3`
      readme and its SAM 3.1 release notes; the `LICENSE` file in `facebook/sam3.1`
- [x] install the official SAM 3 code in a local `.venv` with `uv` (never Homebrew; SPEC facts)
- [x] download the weights with `uvx hf download` into the Hugging Face cache (not the repo, not git)
- [x] pull 3 frames from one ReefCheck video with ffmpeg into `<data_dir>/sam_test/`
- [x] run text prompts ("sea urchin", "purple sea urchin", "red sea urchin", "sea star",
      "sunflower sea star", "sea anemone", "fish") on each
      frame: first on the Apple GPU (MPS), then CPU if MPS fails; at full size with tiles and at
      one reduced size
- [x] on one frame, also test a point prompt (one click on an urchin) and a box prompt, as
      the "add a missed object" step in label review needs them
- [x] record per run: device, seconds per frame (first frame not counted), peak memory,
      masks found; and any error text in full
- [x] write one local report page, `<data_dir>/sam_test/report.html`: each frame with its masks
      drawn, and a table of the numbers. No internet assets.
- [x] if SAM 3.1 will not run: try `facebook/sam3` the same way, then stop and report the options
      (SAM 3, CPU only, another machine). Time box: one working day.
**Outcome (2026-11-24):** both run on the Apple GPU; defaults are SAM 3 + "urchin" (DESIGN D10, Bench results).
**Demo (Nick):** after `uvx hf download facebook/sam3 sam3.pt config.json LICENSE` (once),
`uv run sam-test --video "<one ReefCheck video>"` runs the test and opens
`report.html` in the browser: 3 frames with coloured masks on the urchins, and a table showing
device, seconds per frame and memory. A failed run still produces the page, with the error shown.
**Needs:** S0; Nick logged in to Hugging Face (done 2026-11-24); the SSD plugged in.

### S2 — Frames (ingest, clips, extract, review; no SAM)  `[ ]`
**Goal:** spec stages 1–4 work in the browser on Nick's ReefCheck videos. Design: DESIGN §Planned for S2.
One draft PR per bite; each bite ends with something Nick can click.
Every item of `docs/source/mvp_spec.md` §Stage details 1–4 belongs to the matching bite, including:
ingest takes picked files or a folder (with subfolders), not only `video_roots`; default sampling
every 2 s and cap 300; JPEG quality ≥ 95; counts add up to the live count, or to the cap if the cap
applied; grid from 8 per row down to 1 (default 6); thumbnails never cropped, click for full size;
the review stage can be skipped. A test proves source videos are unchanged after every stage
(SPEC constraint 4).

**Bite 1 — Skeleton + ingest** (split in two PRs, Nick 2026-11-24: 1a skeleton, 1b ingest; 1a: PR 3 merged as `b00be85`, 1b: PR 4 merged as `4d11e30`, Nick ran both tests)
- [x] plan nibble: re-read the prior art (SPEC §Background) and say what is reused (Nick approved 2026-11-24)
- [x] 1a: `uv run app` starts the app on `127.0.0.1` and opens the dashboard; a missing or wrong
      `config.local.toml` path shows a clear message, not a stack trace
- [x] 1a: create / open a project under `<data_dir>/projects/<name>/` (`project.json`, `state.sqlite`)
- [x] 1b: ingest: list videos under `video_roots` with duration, resolution, fps, codec, size (ffprobe);
      skip sidecar (`.LRV`, `.THM`) and `._` files; group GoPro chapters; warn if a video cannot be read.
      Pick files or a folder (the app's own folder browser, starting at `video_roots`, plus a
      pasted path). Same name + same size = one video, shown as "also in: <folder>" (Nick 2026-11-24)
- [x] 1a: dashboard shows every stage and its status (1b fills in Ingest)

**Bite 2 — Clip picker** (split in two PRs, Nick 2026-11-24: 2a player, 2b marks; copies 960 px wide; 2a: PR 5 merged as `9b296cd`, 2b: PR 6 merged as `c730e20`, Nick ran both tests)
- [x] 2a: low-res H.264 playback copy per video, made with ffmpeg into the project folder (GoPro `.LRV` used if present),
      in the background with a progress bar (the small job interface bite 3 reuses)
- [x] 2a: player with timeline and thumbnail strip (one frame every ~5 s, made from the copy)
- [x] 2b: in / out marks with `I` / `O`, optional note; adjust and delete; saved to `clips.json`
- [x] 2b: marked time per video and the number of frames extraction will produce

**Bite 3 — Extract frames** (split in two PRs, Nick 2026-11-24: 3a extraction, 3b filters; DESIGN D17)
- [x] plan nibble: done 2026-11-24 (measurements in DEV_LOG); Nick approved
- [x] 3a: sampling plan (every N seconds inside clips, whole video if none), cap spread evenly, live count
- [x] 3a: frames pulled from the full-resolution original; named `<video-stem>_<path-code>_<timestamp-ms>.jpg`,
      JPEG quality 95 (Pillow); source video and timestamp recorded per frame; upright for videos
      with rotation metadata; frames from earlier settings moved to `dropped/unplanned/` (QE PR 7)
- [x] 3a: runs in the background with a progress bar (`app/jobs.py`); counts shown
- [ ] 3b: blur filter per video (drop the blurriest X % of each video, start 10 %) with a slider and example
      kept / dropped frames; near-duplicate filter (dHash distance to the previous kept frame) with a slider
- [ ] 3b: dropped frames moved to `dropped/blurry/` and `dropped/duplicate/` (never erased; Apply moves them back
      and forth); kept + dropped-blurry + dropped-duplicate add up to the live count (or the cap)
      (QE PR 7 watch: `extract.run` must not re-decode a planned frame whose file sits in `dropped/blurry/`
      or `dropped/duplicate/`, nor set it back to kept)

**Bite 4 — Review frames**
- [ ] grid with size slider, full-screen view (`F`), arrow keys
- [ ] `D` delete (moves the file to `rejected/`), `K` keep, `Z` undo

**Bite 5 — Stills + resume**
- [ ] ingest also accepts folders of still images; they skip clips and extraction
- [ ] each stage resumes where it left off and can be re-run without redoing earlier stages

**Demo (Nick):** `uv run app`, browser opens `http://127.0.0.1:8000`. Pick 2 ReefCheck videos, mark
a clip with `I`/`O`, extract: the kept and dropped counts add up to the preview count. Delete one
frame with `D`, undo with `Z`, delete it again. Quit and restart: the dashboard shows the four
stages where you left them, and the deleted frame is in the project's `rejected/` folder. Then add
a folder of still images: they appear in frame review.
**Needs:** S1 (done).

### S3 — SAM on single frames  `[ ]`
**Goal:** spec stage 5 without job control: class settings file, prompt preview, tiling, overlap rules.
- [ ] `classes.yaml` loaded and shown in the UI
- [ ] urchins: one SAM prompt, "urchin"; its masks are saved as urchins with no type yet (D18)
- [ ] kelp: try prompts for kelp shoot (object) and kelp (area) on frames from a restoration-site
      video; prompts are untested (DESIGN D11)
- [ ] prompt preview on 3 sample frames with masks drawn per class
- [ ] tiling and overlap rules; one result file per frame in `sam_raw/`, recording prompt and settings
- [ ] re-run only what changed; reviewed masks kept and marked "from older settings"
**Demo (Nick):** same start command. On the Labeling page press "Preview": 3 frames show coloured
masks per class. Add a class to `classes.yaml`, run again: only the new class is labeled.
**Needs:** S2; answers to SPEC Q2 and the SAM part of Q10.

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
- [ ] urchin type: `P` / `R` tags the selected urchin purple or red; a frame with an untagged
      urchin cannot be accepted; running totals count purple and red separately (D18)
- [ ] export: `export/images/`, `detector_coco.json`, `seg_masks/` + `classes.json`, `splits/`
      (by site and dive date), `frame_metrics.csv`
- [ ] export checks shown in the UI (COCO load, mask sizes and ids)
- [ ] `THIRD_PARTY_LICENSES` file with the SAM License text
- [ ] README: setup, SAM access steps, 10-minute walkthrough
**Demo (Nick):** review 5 frames, tag their urchins with `P` / `R`, add one missed urchin by clicking it, press "Export". The Export
page shows the COCO load check PASS, mask sizes matching their images, and `frame_metrics.csv`
values matching those 5 frames. All acceptance boxes in `docs/source/mvp_spec.md` are ticked.
**Needs:** S4; answers to SPEC Q9, Q11.

---

## Icebox (captured, not scheduled)

- "Suggest clips" and "Pre-screen with SAM" buttons: left out of S2; decide after S3 (a tiled SAM pass costs about a minute per frame).
- Later MVP milestones (train, quantize, bench test, quality gate, bake-off, Iridium packet): not this repo's Build 1.
