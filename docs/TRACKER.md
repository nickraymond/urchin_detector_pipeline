# TRACKER.md — Sprint Ladder & Rules

*The agent entry point. Newest state lives here.*
*Last updated: 2026-11-25 · Owner/gate: **Nick***

---

## Rules for Agents (READ FIRST, EVERY SESSION)

1. **Read this whole document cover-to-cover first, every session.** Then skim
   `docs/SPEC.md` and `docs/DESIGN.md`. Read the top ~3 entries of
   `docs/DEV_LOG.md`.
2. **Take small code bites.** One TODO at a time, target ~300 LoC. If SPEC.md
   is too thin to inform the bite, the planning lead makes a Lead call
   (CLAUDE.md, "How this team works"); Nick is asked only about the items listed there.
3. **Four nibbles per bite (no Nick gate in any of them):**
   1. **Plan** — at the top of the draft PR; QE checks it against SPEC / TRACKER / DESIGN.
   2. **Code + unit tests.**
   3. **Self-test** — the builder runs the bite in the browser and writes in the PR what it saw.
   4. **PR to QE** — the planning lead merges on QE PASS.
4. **Feature branch for all new work** — `sprint/<n>-<name>`, PR to `main`. Never
   commit to main.
5. **Every sprint ends with one test build Nick runs** — the sprint's Demo
   section, handed over by the lead with a click-through checklist and the
   sprint's Lead calls. His feedback becomes TRACKER items.
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

### S2 — Frames (ingest, clips, extract, review; no SAM)  `[x]`  (demo passed: Nick 2026-11-25 12:09 PM, builder chat: "the demo passes"; last code PR 15 merged as `4fad379`, docs PR 17 as `106f867`)
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

**Bite 3 — Extract frames** (split in two PRs, Nick 2026-11-24: 3a extraction, 3b filters; DESIGN D17; 3a: PR 7 merged as `f9f14b2`, Nick ran the test; 3b: PR 9 merged as `1284232` after QE PASS on `5a72243`, DESIGN D19; Nick's check is in the S2 test build)
- [x] plan nibble: done 2026-11-24 (measurements in DEV_LOG); Nick approved
- [x] 3a: sampling plan (every N seconds inside clips, whole video if none), cap spread evenly, live count
- [x] 3a: frames pulled from the full-resolution original; named `<video-stem>_<path-code>_<timestamp-ms>.jpg`,
      JPEG quality 95 (Pillow); source video and timestamp recorded per frame; upright for videos
      with rotation metadata; frames from earlier settings moved to `dropped/unplanned/` (QE PR 7)
- [x] 3a: runs in the background with a progress bar (`app/jobs.py`); counts shown
- [x] 3b: blur filter per video (drop the blurriest X % of each video, start 10 %) with a slider and example
      kept / dropped frames; near-duplicate filter (dHash distance to the previous kept frame) with a slider
- [x] 3b: dropped frames moved to `dropped/blurry/` and `dropped/duplicate/` (never erased; Apply moves them back
      and forth); kept + dropped-blurry + dropped-duplicate add up to the live count (or the cap)
      (QE PR 7 watch: `extract.run` must not re-decode a planned frame whose file sits in `dropped/blurry/`
      or `dropped/duplicate/`, nor set it back to kept)

**Bite 4 — Review frames** (PR 14 merged as `ee8f8e3` after QE PASS on `b4c956f`; DESIGN D20)
- [x] grid with size slider, full-screen view (`F`), arrow keys
- [x] `D` delete (moves the file to `rejected/`), `K` keep, `Z` undo
- [x] "Show dropped" toggle: blurry, duplicate and rejected frames with their status; `K` makes them `kept_hand`, which the filters leave alone (D20)

**Bite 5 — Stills + resume** (PR 16 merged as `537f9ff` after QE PASS on `82116e7`; DESIGN D21)
- [x] ingest also accepts folders of still images; they skip clips and extraction
- [x] each stage resumes where it left off and can be re-run without redoing earlier stages

**Demo (Nick):** `uv run app`, browser opens `http://127.0.0.1:8000`. Pick 2 ReefCheck videos, mark
a clip with `I`/`O`, extract: the kept and dropped counts add up to the preview count. Delete one
frame with `D`, undo with `Z`, delete it again. Quit and restart: the dashboard shows the four
stages where you left them, and the deleted frame is in the project's `rejected/` folder. Then add
a folder of still images: they appear in frame review.
**Needs:** S1 (done).

### S3 — SAM on single frames  `[x]`  (demo passed: Nick 2026-11-25, lead chat: "S3 passes"; last code PR 28 merged as `49e1db8`)
**Goal:** spec stage 5 without job control: class settings file, prompt preview, tiling, overlap rules.
Design: DESIGN §Planned for S3; Lead calls D22–D27 (they answer SPEC Q2 and the SAM part of Q10).
One draft PR per bite, about 300 lines each.

**Bite 1 — Classes** (PR 20 merged as `373521d`)
- [x] `app/sam/default_classes.yaml` with the starting classes, fixed ids and settings (DESIGN §Planned for S3);
      copied into a project as `classes.yaml` on first use
- [x] urchins: one SAM prompt, "urchin", shared by purple and red urchin through `sam_group: urchin`; its masks
      have no class id until tagged; changing it re-runs both classes (D18)
- [x] checks before any save or run (ids append-only, D6; groups; `seg_as`); errors name the line
- [x] Labeling page: classes table with colours, edit `classes.yaml` in the browser, Reload (D25);
      dashboard row "SAM labeling"; PyYAML a direct dependency with its licence read
- [x] one EXIF-orientation helper for every pixel read (D27); fix the review-grid thumbnails, which show a
      rotated still sideways today (QE PR 19)

**Bite 2 — SAM runner and results files** (PR 21 merged as `c24dba4`)
- [x] SAM in its own worker process (D9), reusing `app/sam_test/device_patch.py`; one model load per job
- [x] tiling above 2000 px, merge to frame coordinates, tile-edge duplicates removed (D23); minimum mask area
- [x] one `sam_raw/<frame>.json` per frame, written after each frame, also with zero masks; records prompts,
      settings and a settings key per run unit
- [x] tests with a fake predictor (no torch, weights or real frames); one real-SAM run in the self-test

**Bite 3 — Prompt preview** (PR 22 merged as `8f3ea1a`; defaults D28, D29)
- [x] "Preview" on 3 sample frames (D26) shows masks coloured per class, a toggle per class, counts,
      seconds per frame and peak memory
- [x] kelp: try prompts for kelp shoot (object) and kelp (area) on frames from a restoration-site video;
      put the best in the default file (D11)
- [x] measure for D24: mask areas of real vs false masks → proposed minimum mask area; re-check D10 on at least
      6 frames from at least 2 videos (urchins kept, seconds per frame, tiles vs whole frame); numbers into
      DESIGN §Bench results

**Bite 4 — Run, re-run only what changed, overlap rules** (PR 24; QE fixes: ids up to 254, D31)
- [x] "Run" labels every kept frame through `app/jobs.py`, with an estimate from the preview (D26); Stop, and
      Run again carries on
- [x] re-run only what changed: per run unit and frame, by settings key; before Run the page lists the classes
      that will run and those unchanged; reviewed masks kept and marked "from older settings"
- [x] `app/sam/overlap.py` with the rules of D22, unit-tested; a "final class map" toggle in the preview
- [x] watch (QE PR 16, PR 19): stills with EXIF orientation: SAM, masks and pages use the same upright image (D27);
      S5's export images must be upright too (still open for S5)

**Bite 5 — Large-image viewer on the Labeling page** (Nick, 2026-11-25, SPEC Q16; PR 27)
- [x] clicking a preview frame opens it large on the same page (no new tab), with < > buttons and arrow keys
      through the preview frames, Esc to close, zoom and pan: S2's `static/viewer.js`, reused, not copied
- [x] a class legend along the bottom of the viewer, with each class's colour and mask count. Clicking a class
      turns its masks on or off; a second control shows only that class, so "urchin" masks stand out from
      everything else. The legend applies to the large view and the preview tiles alike
- [x] the legend says why some masks have a box: objects (counted things) are boxed, areas (kelp, rock) are not

**Bite 6 — Class editor form** (Nick, 2026-11-25: the YAML text box invites typing errors; replaces D25's text box; PR 28)
- [x] a form instead of the text box: one row per class with name, kind (object / area as a choice), prompts as
      chips (add / remove), group and "pixels count as" as drop-downs of valid values, colour swatch; an
      "Add class" button that gives the next free id; settings as labelled number fields
- [x] ids are shown and never typed. A class with results keeps its name (D6), and the form says why
      renaming is locked; removing it keeps its saved results (DESIGN §Planned for S3). Every change is
      checked before Save with the same checks as `classes.py`, and nothing invalid is saved
- [x] the page's Save still writes `classes.yaml` (so a text editor plus Reload keeps working); test: no form
      input can produce a file that fails the checks

**Demo (Nick):** same start command, a project of about 5 kept frames. On the Labeling page press "Preview":
3 frames show coloured masks per class, with counts and seconds per frame; "final class map" shows one colour
per pixel, kelp on top of rock. Press "Run". Then add a class on the page and press "Run" again: the page says
only the new class will run, and afterwards the other classes' results are unchanged.
Click a preview frame: it opens large on the page; < > move to the next frame; click "urchin" in the legend
to show only urchin masks. Add a class with the form (no typing of YAML) and Save.
**Needs:** S2 (done); SPEC Q2 and the SAM part of Q10 answered by Lead calls D22–D24 (Nick can reverse them).

### S4 — Job control  `[~]`
**Goal:** long jobs run in a background worker with estimate, pause/resume, limits and a schedule.
Built as its own module with no labeling-specific code (DESIGN D4). Design: DESIGN §Planned for S4;
Lead calls D32–D38 (they answer the memory-guard part of SPEC Q10). One draft PR per bite, about 300 lines each.

**Bite 1a — Job module core** (D32, D33; PR 32)
- [x] `app/jobctl/`: one queue for the whole app, one job at a time; job records in `<data_dir>/jobs/jobs.sqlite`
      (kind, params, state, progress, times, stop reason); a job is a worker process speaking `@@ {json}` lines and
      reading `pause` on stdin; worker helpers (`app/jobctl/worker.py`); nothing in it imports `app.stages` or
      `app.sam` (a test checks)
- [x] Pause / Resume / Cancel and "interrupted" after an app quit, tested end to end with a fake job kind (a tiny
      worker that writes one file per item): resume never repeats or skips an item

**Bite 1b — SAM on the job module; Jobs page** (D33, D34; PR 33)
- [x] SAM Preview and Run go through it; the SAM worker's `frame` event becomes `item` and it answers `pause`;
      Pause finishes the frame in progress and ends the worker (model unloaded); Resume starts a new worker on the
      frames still missing; Cancel keeps finished frames. S3's Stop becomes Pause
- [x] progress per frame from the artifacts (`sam_raw/`): resume never repeats or skips a frame (test with the fake predictor)
- [x] app quit or crash mid-job: on the next start the job shows as "interrupted", and the dashboard offers Resume
- [x] a Jobs page (`/jobs`): every job with state, progress, Pause / Resume / Cancel; linked from the dashboard

**Bite 2 — Extraction and playback copies on the job module; `app/jobs.py` retired** (D32; PR 34)
- [x] frame extraction runs as a job-module worker process with Pause / Resume / Cancel (skips frames on disk, as today)
- [x] playback copies and thumbnail strips queue on the same module (no pause needed); `app/jobs.py` is removed
- [x] the Extract and Labeling pages show the job's state from the module; one job at a time across all stages

**Bite 3 — Estimate and remaining time** (D35; PR 35)
- [x] "Estimate" on the Labeling page: 1 warm-up frame (not counted) + 5 frames from those Run would label, at the
      current settings and limits; shows seconds per frame, frames, total time and finish clock time (now); the 6
      frames' results are kept, so Run does not redo them
- [x] during any job: remaining time from the rolling average of the last 20 items

**Bite 4 — Resource limits** (D36; PR 36)
- [x] settings panel on the Jobs page: device (auto / Apple GPU / NVIDIA GPU / CPU), CPU threads (default half the
      cores), low priority (`nice`, on by default), pause between frames (default 0 s), memory guard, batch size 1
      (shown, fixed); each job records the limits it ran with
- [x] memory guard: the app checks free memory every 2 s while a job runs and pauses it, saying why, below the
      threshold; the builder measures the guard's own free-memory figure (D36) during a tiled 4K and a whole-frame 5.3K run and proposes the
      default in the PR (D36)
- [x] a CPU fallback on the Apple GPU shows as a warning on the job

**Bite 5a — Schedule** (D37; PR 37)
- [x] "Start at" and optional "Stop by" on extraction and SAM Run; a queued job shows its start time, Start now
      and Cancel; the schedule survives an app restart; "Stop by" pauses the job and says so; the Estimate's finish
      clock time counts from the scheduled start (moved here from bite 3, QE PR 35)

**Bite 5b — Keep awake, log and morning summary** (D38; PR 39)
- [x] "Keep laptop awake" toggle (macOS `caffeinate`) offered when a long or scheduled job starts, released when no
      such job is left; reminder: plugged in, lid open; a warning when the laptop runs on battery
- [x] one log file per job; a summary per job (items done, counts per class including urchins, errors, time taken,
      why it stopped); the dashboard shows the last finished job's summary ("morning summary")

**Demo (Nick):** press "Estimate" → seconds per frame and a finish time. Start a run and keep
using the laptop (browser, a video): it stays usable with the thread cap on. Pause, quit the app,
start it again, resume: the frame counter continues with no frame repeated or skipped. Schedule a
start 2 minutes ahead with "Keep laptop awake" on, wait, read the summary page.
**Needs:** S3 (done).

### S5 — Review labels and export  `[ ]`
**Goal:** spec stage 6 and all outputs; the spec's acceptance criteria pass on Nick's videos.

**Anemone prompt check** (Nick, 2026-11-25: SAM calls urchins "anemone"; a bench like S3 bite 3; run SAM only when
no other SAM run is going)
- [ ] on frames where an urchin shows as an anemone, see whether SAM also finds it as an urchin (legend "only")
- [ ] try other anemone prompts (e.g. "anemone", "sea anemone tentacles") and say which keeps real anemones and drops urchins
- [ ] whether a confidence per class is needed (today one value, 0.4, for all); prompt and confidence choices are
      Lead calls the lead logs. A rule that urchin beats anemone whatever the scores would contradict the spec's
      "keep the higher score" (D22 rule 1, D31): if the bench argues for it, it goes to Nick as a question
- [ ] best prompt into `app/sam/default_classes.yaml`; numbers into DESIGN §Bench results

**Label review and export**
- [ ] label review uses the app's one large-image viewer and its class legend (SPEC Q16, S3 bite 5)
- [ ] label review page: class change, delete mask, add a missed object by click or box, accept
      frame, split check, fully-checked flags, running totals
- [ ] urchin type: `P` / `R` / `B` tags the selected urchin purple, red or black; a frame with an untagged
      urchin cannot be accepted; running totals count each type separately (D18, D39)
- [ ] black urchin (SPEC Q18): a new class with the next free id in the default `classes.yaml`, in the
      `urchin` group; existing projects can add it from the class form. The id never changes; a later rename
      keeps the id and must be possible even after results exist (Lead call D39, lifting S3's name lock for renames)
- [ ] fast tagging (Nick, 2026-11-25: most urchins are purple, find the outliers): `Tab` jumps to the next
      untagged urchin and zooms the viewer onto it with some scene around it; a close-up panel shows the
      selected urchin enlarged; "Tag all remaining as purple" tags every untagged urchin on the frame at once
- [ ] export: `export/images/`, `detector_coco.json`, `seg_masks/` + `classes.json`, `splits/`
      (by site and dive date), `frame_metrics.csv`
- [ ] export checks shown in the UI (COCO load, mask sizes and ids)
- [ ] `THIRD_PARTY_LICENSES` file with the SAM License text
- [ ] README: setup, SAM access steps, 10-minute walkthrough
**Demo (Nick):** review 5 frames, tag the outlier urchins with `R` / `B` and the rest with "Tag all remaining as purple", add one missed urchin by clicking it, press "Export". The Export
page shows the COCO load check PASS, mask sizes matching their images, and `frame_metrics.csv`
values matching those 5 frames. All acceptance boxes in `docs/source/mvp_spec.md` are ticked.
**Needs:** S4; answers to SPEC Q9, Q11.

---

## Icebox (captured, not scheduled)

- "Suggest clips" and "Pre-screen with SAM" buttons: left out of S2; decide after S3 (a tiled SAM pass costs about a minute per frame).
- Re-check the kelp prompts (D29) and urchin defaults on more restoration-site footage once the full pipeline
  works and more video is available (Nick, 2026-11-25: only the SSD's videos for now).
- Later MVP milestones (train, quantize, bench test, quality gate, bake-off, Iridium packet): not this repo's Build 1.
