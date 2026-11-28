# TRACKER.md — Sprint Ladder & Rules

*The agent entry point. Newest state lives here.*
*Last updated: 2026-11-27 · Owner/gate: **Nick***

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

### S4 — Job control  `[x]`  (demo passed: Nick 2026-11-25 10:13 PM, lead chat: "I've tested everything in the check list and it passes"; feedback bites 6, 6b, 6c merged as PRs 41, 42, 45; last `57ebe81`)
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

**Bite 6 — Nick's test-build feedback** (D34a, D36b; PR 41)
- [x] Run panel in this order: how long it takes (total, s / frame and its source), when it finishes, then a
      class | frames-to-run table; the Estimate, Preview and job status lines follow the same order; hints shortened
- [x] Cancel stops at once, also while a job is pausing (the frame in progress is dropped, finished ones kept); a
      pausing job says what it waits for ("finishing frame 2 of 40, about 1 min")
- [x] a harmless CPU fallback (`aten::_assert_async`) goes to the job log only; any other stays a warning, worded
      "one step runs on the CPU (slower); SAM itself runs on the Apple GPU"

**Bite 6b — Nick's test-build feedback, items 4–8** (D36c; PR 42)
- [x] Resume easy to find: buttons next to the state on the Jobs page; a paused, pausing or interrupted job shows a
      banner with Resume at the top of the Jobs, dashboard, project, Labeling and Extract pages
- [x] while a job runs: a progress bar, a moving "running" sign and "about N min left, done by H:MM PM"
- [x] pages stay in sync in place: job boxes poll an HTML fragment every 2 s and swap only what changed; the Jobs
      page's 3 s reload is gone; the page's own buttons follow the job
- [x] a memory-guard pause says why on the job, everywhere; it resumes by itself once twice the limit has stayed free
      for 60 s (D36c); the summary counts guard pauses and auto-resumes
- [x] the project page's Extract and SAM labeling rows show their job (state, mini bar, time left), live

**Demo (Nick):** press "Estimate" → seconds per frame and a finish time. Start a run and keep
using the laptop (browser, a video): it stays usable with the thread cap on. Pause, quit the app,
start it again, resume: the frame counter continues with no frame repeated or skipped. Schedule a
start 2 minutes ahead with "Keep laptop awake" on, wait, read the summary page.
**Needs:** S3 (done).

### S5 — Review labels and export  `[ ]`
**Goal:** spec stage 6 and all outputs; the spec's acceptance criteria pass on Nick's videos.
Design: DESIGN §Planned for S5; Lead calls D40–D46 (they answer SPEC Q9 and Q11). One draft PR per bite, about
300 lines each. Bites 1–2 are backend only (new files), so they can run while S4's last bites touch the pages.

**Carried from S4's test (S4 builder, its bite 6c)** (Nick, lead chat 2026-11-25, 9:55–10:14 PM; PR 45)
- [x] "Labelled so far" on the Labeling page: each frame appears as soon as SAM finishes it, newest first, drawn
      like the preview tiles (overlay, class legend, large viewer with < >), Pause / Cancel next to it; it stays
      after Run ends and lists every labelled frame with its counts per class
- [x] the project page's SAM labeling row shows the labelled frames from disk after a run ("21 of 21 frames
      labelled"), not "not started"

**Bite 1 — Review state** (D40; PR 44; the urchin type is the class id)
- [x] every mask in `sam_raw/` gets a stable `mask_id`; a re-run keeps the ids of masks it keeps
- [x] `state.sqlite` tables for mask review (class, deleted, urchin type, reviewed) and frame review (accepted,
      split check yes / no, classes fully checked); helpers to read a frame's final masks = `sam_raw` + review,
      after the overlap rules (D22, D31); unit tests with synthetic masks
- [x] accept is refused while the frame has an untagged urchin (D18); undo for every review action

**Bite 2a — Export writers** (D41, D44; PR 46)
- [x] `app/export/`: `export/images/` (upright, D27), `detector_coco.json` (object classes, box + outline + class;
      frames with split check "no" left out), `seg_masks/` one PNG per frame + `classes.json` (ids, names,
      colours); rebuilt from scratch each time from `sam_raw` + review state; synthetic tests

**Bite 2b — Splits, metrics and checks** (D41, D43; PR 46)
- [x] `splits/` by site and dive date (D43); `frame_metrics.csv` (source video, timestamp, count and % cover per
      class, largest object %)
- [x] checks as data: COCO loads with pycocotools; every mask PNG matches its image size and uses only ids in
      `classes.json`; metrics match the masks; synthetic tests

**Bite 3a — Label review page** (PR 50)
- [x] a Review page per project: frame list with accepted / to do, the one large viewer and class legend (SPEC
      Q16), objects only / areas only / final class map toggle, zoom slider and full screen
- [x] click a mask to select it; searchable class picker to change its class; `X` deletes; `Enter` accepts; `Z` undo

**Bite 3b — Split check, fully checked, totals** (PR 51)
- [x] split check (touching urchins separated: yes / no) and "fully checked" per class on the frame
- [x] running totals per class (objects and frames, target 150–200 objects per class)

**Bite 4a — Urchin types, backend** (D18, D39; PR 47; split from bite 4 by the builder: the keys need bite 3's page)
- [x] black urchin (SPEC Q18): a new class with the next free id in the default `classes.yaml`, in the
      `urchin` group; existing projects can add it from the class form. The id never changes; a later rename
      keeps the id and must be possible even after results exist (Lead call D39, lifting S3's name lock for renames)
- [x] tagging helpers (`app/review/tagging.py`): types and keys, next untagged, zoom rectangle, tag, tag all remaining

**Bite 4b — Urchin types and fast tagging on the Review page** (D18, D39; PR 54)
- [x] `P` / `R` / `B` tags the selected urchin purple, red or black; running totals count each type
- [x] fast tagging (Nick, 2026-11-25: most urchins are purple, find the outliers): `Tab` jumps to the next
      untagged urchin and zooms the viewer onto it with some scene around it; a close-up panel shows the
      selected urchin enlarged; "Tag all remaining as purple" tags every untagged urchin on the frame at once

**Bite 5a — Add a missed object, backend** (D46; PR 52; split by the builder: the clicks need bite 3's page)
- [x] SAM 3's click / box prompt on a crop around the spot (`app/sam/add.py`, `app/sam/add_worker.py`), job kind
      `sam_add` on the job module; saved under the unit `added`, marked added by hand and reviewed

**Bite 5b — Add a missed object on the Review page** (D46; PR 55)
- [x] click on it or drag a box; SAM returns a mask for that spot (S1: SAM 3's click and box prompts work);
      pick its class; saved like any mask, marked added by hand; runs through the job module

**Bite 6a — Export page** (D43; PR 56; split by the builder)
- [x] an Export page: Export button (a job), the checks from bite 2 shown PASS / FAIL with what failed, counts per
      split and per class, a warning when the splits cannot be fair (D43)

**Bite 6b — Licences, README, acceptance** (D42; PR 57)
- [x] `THIRD_PARTY_LICENSES` with the SAM License text; README: setup, SAM access steps, 10-minute walkthrough (PR 57)
- [~] run every acceptance box of `docs/source/mvp_spec.md` on Nick's videos and tick it with its evidence (PR 57:
      9 of 10 ticked with their evidence; the `frame_metrics.csv` spot check on 5 frames is Nick's, in the S5 demo)

**Fix — Missing urchin type in older projects** (D50; found in Nick's S5 test)
- [x] a project made before D39 has no black urchin, so no B key: the Review page offers "Add black urchin" (appended,
      new id); nothing else changes

**Bite 7 — SAM quality: tile seams on areas, anemone prompts** (D45, D48, D49; PR 49; needs the GPU: run SAM only when no
other SAM run is going)
- [x] area masks (rock, kelp) end in straight lines where tiles meet (Nick, 2026-11-25, S4 test frames): compare
      areas on the whole frame (one extra image encoding per frame) with tiles on the same frames; pick by the
      overlays, seconds per frame and peak memory; numbers into DESIGN §Bench results
- [x] anemone (Nick, 2026-11-25: SAM calls urchins "anemone"): on frames where an urchin shows as an anemone,
      see whether SAM also finds it as an urchin; try other anemone prompts (e.g. "anemone", "sea anemone
      tentacles"); whether a confidence per class is needed (today 0.4 for all). Prompt and confidence choices
      are Lead calls; a rule that urchin beats anemone whatever the scores would contradict the spec's "keep
      the higher score" (D22 rule 1, D31): if the bench argues for it, it goes to Nick as a question
- [x] best settings into `app/sam/default_classes.yaml`

**Demo (Nick):** review 5 frames, tag the outlier urchins with `R` / `B` and the rest with "Tag all remaining as
purple", add one missed urchin by clicking it, press "Export". The Export page shows the COCO load check PASS,
mask sizes matching their images, and `frame_metrics.csv` values matching those 5 frames. All acceptance boxes in
`docs/source/mvp_spec.md` are ticked. Also re-check S4's feedback bites (Run panel, Cancel, progress, Resume,
pages in sync, memory-guard message, project page, "Labelled so far").
**Needs:** S4 (demo passed); SPEC Q9 and Q11 answered by Lead calls D41, D42 (Nick can reverse them).

### Build 2 — what runs where, then train, compare and bench test (S6–S11)
Source: `docs/source/build2_spec.md` (Nick's answers of 2026-11-26 and 2026-11-27 are in it). Order (Nick,
2026-11-27): first what each camera can run for each kind of model, then training and the accuracy comparison.
The lead writes each sprint's design in DESIGN before its first bite and splits bites to about 300 lines; S7 and
S9–S11 below are outlines until then. Nothing here touches `app/review/` or `app/export/` beyond reading the export.

### S6 — What runs where, on the laptop  `[~]`
**Goal:** the first version of the "what runs where" table (spec §First result): every candidate architecture of
the three kinds, untrained, converted to 8-bit and put through each camera's compiler. No hardware, no training,
no labeled data. Reuse first: the export and compile scripts of `ADIN_SPI_OpenMV/ml/` (`yolox_urchin/export.py`,
`compile_model.sh`) and its compile-gate method (`ml/compile_gate_report.md`).

**Bite 1 — Prior art, toolchains, licences (plan nibble + setup, no app pages)** (D51–D55; DESIGN §Planned for S6)
- [x] re-read the earlier export and compile scripts and say what is reused and how it is brought in
- [~] set up as documented, versions and licences recorded: the PyTorch → 8-bit route, ST Edge AI Core 4.1.0 (Nick said yes to
      ST's licence terms, 2026-11-27: read them first, save a copy in the data folder, stop on anything unusual), Arm Vela (Apache-2.0), and the Pi runtime candidates on the laptop
      *Done 2026-11-27: the 8-bit route (`convert-venv`), Vela 5.0.0 native, LiteRT on the laptop. Open: ST Edge AI
      Core 4.1.0 is not installed (the builder session was refused permission to run ST's installer; its terms,
      SLA0044, are read and saved in `<tools_dir>/licences/` and `<data_dir>/fit/licences/`). S6 uses the known-good 4.0.0 in the container (D52); ncnn and ONNX Runtime
      for the Pi are set up in bite 3c.*
- [x] known-good first: reproduce the earlier YOLOX-Nano result on both compilers (AE3 one `ethos-u` block; N6
      about 117 + 2 + 0 of 119, the earlier trained export; untrained it was 116 + 2 + 0 of 118) before changing anything
- [x] decide and log in DESIGN: where these tools' Python packages live (the app's PyTorch is pinned to 2.10.0,
      D7), the `models/` layout, what a "recipe" holds

**Bites 2a, 2b, 2c — Candidate list, one bite per kind**
- [x] 2a-i (split by the builder): `app/modeldefs/` (D51, D54): a candidate is the `arch` part of a recipe, with
      its folder name (`arch_key`) and checks; YOLOX Nano / Tiny with the plain first layer at 256 and 320 px, five
      classes, built untrained with the three raw output maps (adapted from the earlier `yolox_urchin/model.py`)
- [x] 2a-ii: RF-DETR Nano (detector): licence at the source, what building it downloads or sends (nothing: built
      with the network refused), model definition at 384 and 256 px; its own tool environment and export settings (D56)
- [x] 2b: segmenters (D57): torchvision LR-ASPP and DeepLabV3 on MobileNetV3-Large, and a plain conv net of our
      own in two sizes; 256 px, the five classes plus "other"; built untrained in the app's own tests
- [x] 2c-i: instance segmenters (D58): RF-DETR-Seg Nano (312 px) and YOLOX Nano / Tiny with prototype masks (our
      own small conv one, 256 px)
- [x] 2f (Lead call D71): does ST Edge AI Core 4.1.0 (Nick's install) compile any RF-DETR size for the N6? No:
      full-integer TFLite and quantized ONNX both stop (DESIGN §Bench results, bite 2f); YOLOX Nano gives the same
      117 + 2 + 0 as 4.0.0. A guarded second N6 compiler step (`app/fit/n6_st41.py`): "n" to ST's statistics
      question, network and `~/.stmai` denied, refuses to start otherwise; placement only
- [ ] 2f, page part (not built): show a candidate's 4.1.0 result (`targets/n6_st41/fit.json`) as a second line
      in its N6 row, with a button to run it; today the step can be queued only from code
- [x] 2c-ii: RTMDet-Ins-tiny (Apache-2.0) as a published small conv instance segmenter
      (`app/modeldefs/rtmdetins.py`, D72): mmdetection's pure-Python packages in `mmdet-venv`; built untrained with
      no network; through the real job: runs, partly on the CPU, on the N6 (2 software steps) and the AE3 (1 CPU
      operator), runs on the Pi side (DESIGN §Bench results). SparseInst not tried (time box)
- [ ] the candidate architectures with code licence read at the source and added to `THIRD_PARTY_LICENSES`,
      each as a model definition the app can build untrained at a given input size; anything not permissive
      stays out: 2a detector (YOLOX Nano / Tiny, RF-DETR Nano); 2b segmenter (two or three small ones); 2c
      instance segmenter (RF-DETR-Seg Nano and one or two small conv ones)
- [~] 2d: RF-DETR Atto / Femto / Pico (PML 1.0; evaluation only, SPEC constraint 6): Nick has
      accepted Roboflow's terms on his own account (2026-11-27); the builder confirms the account's plan covers `rfdetr[plus]` and first shows what the package sends out (no frames go
      through it before that); marked "Roboflow licence" in the table
      *2026-11-27 (D59): what it sends out is shown (static read; built with the network refused: 0 attempts) and
      the three sizes are candidates, marked. Open, with Nick through the spec lead: whether his plan covers
      `rfdetr[plus]` (a builder cannot see his account) and the usage reporting of PML section 4.*
- [x] 2e (Lead call D70; the question Nick asked): one retry of the RF-DETR family with one thing changed,
      RF-DETR's own export (onnx2tf 2.4.3) plus a full-integer step, in two new Python 3.12 tool environments;
      `convert-venv` untouched. Result (DESIGN §Bench results, bite 2e): all six convert; none compiles for the N6;
      on the AE3 only Atto compiles and fits, partly on the CPU; on the Pi side the files run under plain LiteRT.
      The route is now the family's 8-bit step in the convert job; the licence text goes with every
      Roboflow-licensed result (PML section 7)

**Bites 3a, 3b, 3c — Convert job (3a the job and the 8-bit step with a fake compiler; 3b the N6 and AE3
compilers; 3c the Pi)**
- [x] 3a: job kind `convert` (`app/fit/convert.py`, `convert_worker.py`): 8-bit step in the model's tool
      environment (`app/fit/to_int8.py`: the earlier route, seeded weights and noise calibration, D55), then the
      camera's compiler module, then `fit.json`; a failed 8-bit step or compile is a result and is not repeated for
      the next camera; Pause ends the 8-bit step cleanly; `tools_dir` setting; tested with a fake 8-bit step and a
      fake compiler
- [x] 3b: the N6 and AE3 compiler modules (`app/fit/n6.py`, `ae3.py`, D52) and their report parsers
      (`app/fit/reports.py`, tested on excerpts of the real reports); `[fit]` settings; self-test: 10 candidates ×
      2 cameras through the job (DESIGN §Bench results)
- [x] 3c: the Pi (`app/fit/pi.py`): the 8-bit file runs under LiteRT on the laptop; operators, any that is not
      built in, file size; the laptop's time is kept as a laptop figure only. ONNX Runtime and ncnn are compared on
      the bench Pi in S7 (each needs its own 8-bit route)
- [ ] job kind `convert` on `app/jobctl/`: one candidate × one camera → 8-bit model → the camera's compiler →
      a fit report as data (converts yes / no; layers on the NPU / hybrid / CPU; file size and memory against
      the board's limits; the compiler's own speed estimate, marked as an estimate; the failing layers by name);
      a failed conversion is a result, not a job error; tested with a fake compiler

**Bite 4 — "What runs where" page**
- [x] 4a (split by the builder): the page `/fit` (`app/fit/table.py`, `pages.py`, `fit.html`): kind × camera, each
      cell's headline from its best candidate, per candidate the layers, sizes against limits, the compiler's
      estimate marked as such, the failing layers or the error; earlier measured numbers as "earlier work" with
      their date; "Run all" queues every missing cell once; Run / Run again per candidate; the Roboflow-licensed
      sizes always marked; linked from the dashboard
- [x] 4b: "stored together" on the page: for each pair and all three of the best candidates per camera, their
      files against the model area beside the vendor's models (N6, AE3) or against the Pi's memory, and the room
      left; the AE3 time-box note
- [x] "Run again" on one camera cleared the candidate's shared 8-bit file while the other cameras' cells kept
      results from the old file (QE PR 80 nit 4): it now clears every camera's result of that candidate and
      queues the other cells that had a result again behind it (PR 85)
- [x] a LiteRT fault on the Pi step (runtime missing or changed) is a job error with a fix, not "does not run on
      the Pi" (QE PR 79 nit 1; PR 85)
- [x] 3d (found by the builder; D51 promised it): "Set up the tools" as a job with a button on `/fit`
      (`app/fit/setup.py`, `setup_worker.py`): the four tool environments from `app/fit/toolchain/` with `uv`, the
      YOLOX checkout at its commit with `git`; what is ready is skipped; the N6 compiler's pieces from Nick's
      earlier work are only checked. Tested with stand-ins for `uv` and `git`
- [ ] the table: kind × camera; each cell lists its candidates with the fit report and a headline (runs /
      runs, partly on the CPU / does not run / not tried); earlier measured numbers shown as "earlier work" with
      their date; "Run all" queues every missing cell
- [ ] several models on one camera, laptop part: for each pair and all three of the best candidates per
      camera, whether their files fit side by side in the model area (N6, AE3) or in storage and memory (Pi), and how
      much room is left
- [ ] the AE3 column is under the one-day time box: what is not converted in that time is recorded with its reason

**Demo (Nick):** open "What runs where" and press "Run all": the table fills, cell by cell, for the three kinds
on the N6, the Pi and the AE3. Click a cell: per candidate, the layers on the NPU and on the CPU, sizes against
the limits, and for a failure the layers that stopped it.
**Needs:** the Build 2 spec on main.

### S7 — What runs where, on the boards  `[ ]`
*State 2026-11-27 (Build 2 lead 2): the rig test session owns the rig; its backup passed QE with three declared
gaps (below). Evidence: `~/NereusData/urchin_detector_pipeline/rig_backup/2026-11-27/` (never in git). No S7 code
is started.*
**Goal:** the table complete with measured numbers: each candidate that compiled is loaded on its board and
timed. The team's test session runs it on Nick's Pi Zero 2W bench rig (IMX708, N6, AE3), which it owns from
Nick's hand-over; builders and QE use a fake board. Nick is asked only for physical actions.
Design: DESIGN §Planned for S7; Lead calls D81–D86. One draft PR per bite, about 300 lines each. Builders test
against the fake board only; the rig test session runs each board's first card after that board's bite is merged.

- [x] **rig backup first, before any change** (SPEC constraint 9): done 2026-11-27 by the rig test session, QE 7
      PASS with three declared gaps: the AE3's RECOVERY partition (its bootloader refuses it), no raw card image (a
      software-only backup instead), and the restore steps not rehearsed. A rehearsal plan waits for Nick (SPEC
      "Waiting for Nick")
- [ ] the raw card image of the Pi, now that the media are deleted (SPEC Q33; constraint 9 as written), checked
      like the rest; **the first rig change waits for it** and for QE's review of the first card

**Bite 1 — Bench units, the `Board` interface, the fake board, the rig log** (D81)
- [ ] `[bench]` in `config.local.toml` (ssh host alias, camera kinds, `field_units` refused by name; nothing in git);
      `app/bench/`: `Board` (info, load, time, load-several, reset) and `FakeBoard` (timings, memory, failures on
      request); any unit not listed is refused, tested
- [ ] `<data_dir>/bench/rig_log.jsonl`: every change appended before and after, with its undo; a "Rig log" page

**Bite 2 — Test card and the `bench` job** (D83, D84)
- [ ] the card as data and a page (question, units, models and sha256, frames, windows and N, metric, pass bar,
      stop rules, cost, written by, reviewed by); Run is refused without a saved card; the card is saved with the run
- [ ] job kind `bench` on `app/jobctl/`: one card → windows → `bench/<run id>/` (`run.json`, `timing.jsonl`); Pause
      after a window, Cancel keeps finished windows; stop rules as job errors with a recovery hint; tested end to end
      with the fake board

**Bite 3 — The bench agent and the Pi** (D81, D85)
- [ ] the agent (`app/bench/agent/`, plain Python 3): copied to its folder on the Pi over ssh, speaks JSON lines;
      tested on the laptop against a fake transport
- [ ] Pi runs with LiteRT first (same 8-bit file as S6): N windows, `perf_counter_ns` around invoke, `VmHWM`;
      then ONNX Runtime; ncnn time-boxed to half a day; each runtime's licence read at the source before install

**Bite 4 — The N6 route** (D82, D83)
- [ ] ROMFS0 image with the vendor's models plus ours (`build_romfs_n6.sh` method), DFU alt 3, read-back
      verify, `/rom` listing; the camera timing script (2 warm-ups, `ticks_us` around N × `predict`, `gc.mem_free`);
      tested with the fake board and on a ROMFS image file on the laptop
- [ ] first real card (rig test session): YOLOX Nano re-measured against the earlier 10.6 ms before any new model

**Bite 5 — The AE3 route** (D82; under the one-day AE3 time box, S6's compile time counted)
- [ ] `/flash` route with a heap check and an on-board size and sha256 check; ROMFS0 route for larger files with
      our own build script on OpenMV's `mkromfs.py`; first real card: YOLOX Nano against the earlier 26.4 ms

**Bite 6 — Several models on one camera, board half** (DESIGN §Planned for S7)
- [ ] the pairs and the three from S6's "stored together": loaded together (peak memory) or, if not, load, run and
      unload in one script with the whole sequence timed; headline per camera: both at once / one after the
      other / one kind only

**Bite 7 — Power add-on** (D86)
- [ ] without a meter every run shows "not measured"; a meter listed for a unit adds `power.jsonl`, gross and net
      energy per inference and a power-over-time chart; "not validated" until a saved known-load check; fake meter

**Bite 8 — Measured numbers on the page, and the report**
- [ ] each cell shows the median ms and spread, memory, and energy (or "not measured") next to the compiler's
      estimate, linked to its card and run; one-page "what runs where" report: which kinds run on which camera
      and why not, and the several-models headline per camera

**Demo (Nick):** nothing to run at the bench. Open "What runs where": the cells show ms per inference and memory
measured on the N6, the Pi and the AE3, each linked to its test card and run. Open the report: which kinds run on which camera and why not, and whether each camera can hold and run
several models at once or only one kind at a time.
**Needs:** S6; the rig set up by Nick and handed to the test session; SPEC hard constraint 9.

### S8 — Dataset versions and the first detector  `[~]`
**Goal:** spec stages 7 and 8 for one model type already known to run on the cameras (YOLOX, earlier work): freeze a dataset version from a Build 1 export and train a
YOLOX detector on it as a job. Reuse first: the trainer and recipe from `ADIN_SPI_OpenMV/ml/yolox_urchin/`.

**Bite 1 — Prior art and licences (plan nibble, no app code)**
- [x] re-read `ADIN_SPI_OpenMV/ml/yolox_urchin/` (trainer, export, decode) and say what is reused and how it is
      brought in; read the licences of YOLOX's code and of any starting weights at the source; add them to
      `THIRD_PARTY_LICENSES` (DESIGN §Planned for S8; published COCO weights are not offered: licence not verified)
- [x] decide and log in DESIGN: the layout of `datasets/` and what training adds to S6 bite 1's decisions
      (package location, `models/` layout, recipe) (D60, D61)

**Bite 2a — Dataset versions**
- [ ] a Datasets page per project: "Freeze a version" from the current export (a job): manifest with frames by
      reference, splits, class list, counts per class and split, a hash; versions are numbered and never changed
- [ ] a new version keeps every site-and-date group in the split it had in the version before; new groups are
      dealt to keep about 70 / 20 / 10 (the export's own `splits/` may re-deal on each run, D43)

**Bite 2b — What training will see**
- [ ] the page shows what training will see: objects and frames per class and split, classes under the 150–200
      target, frames left out by the split check, classes not fully checked per frame
- [ ] loaders for training: a class not fully checked on a frame is skipped for that frame; split check "no"
      frames are left out of detector data; synthetic tests

**Bite 3 — Outside data and starting weights**
- [ ] a version may add a public source from `~/nereus_ml/datasets/` whose recorded verdict allows commercial
      use (not the DUO set: Nick, 2026-11-27); the page lists each source with its licence and counts; its
      boxes are one class, "urchin", used for pre-training the detector only, never in the test split; the
      folder is a setting in `config.local.toml`
- [ ] starting weights are a recorded choice per run: from scratch, pre-trained on the allowed sources, or
      published weights with a recorded licence

**Bite 4a — Train job**
- [ ] job kind `train` on `app/jobctl/`: a worker process that writes a checkpoint and a metrics line per epoch;
      Pause finishes the step, saves and unloads; Resume continues from the last checkpoint; Cancel keeps
      checkpoints; tested end to end with a tiny fake trainer (no torch)

**Bite 4b — YOLOX trainer**
- [ ] classes: sea stars and urchins only (Nick, 2026-11-27): purple, red and black urchin, sea star, sunflower
      sea star by default; still a setting per run
- [ ] YOLOX-Nano with the plain first layer, tiles as part of the recipe, an augmentation preset that
      includes blur and haze; every run records dataset version,
      recipe, code version, seed, library versions and device

**Bite 5 — Train page**
- [ ] pick a dataset version, model type and size, classes and recipe (defaults with a short "why"); Estimate
      (time a few steps), Run, Start at / Stop by, Keep laptop awake, limits
- [ ] live: loss and validation curves, a few validation frames with boxes drawn, time left; the finished
      model appears in a plain model list

**Fixes found 2026-11-27**
- [ ] the job runner starts before the job kinds are registered (`app/web/server.py` `jobctl`), so a queued or
      due scheduled job at app start can end "no job kind … is registered" (QE 8, reproduced with a 0.3 s delay;
      one `tests/test_schedule.py` failure in PR 97's suite): create it without starting, register, then start
- [ ] the trainer's validation shrinks the whole frame to the canvas (`samples.py` `_scale` with `train=False`),
      a different mode from training and from S9's eval scale (D90): validate at the eval scale with tiles

**Demo (Nick):** on a project with an export, open Datasets and press "Freeze a version": the page shows counts
per class and split. Open Train, pick YOLOX-Nano, press Estimate, then Run: the loss curve moves and validation
frames show boxes. Pause, quit the app, start it, Resume: training continues from the last checkpoint. The
model is in the list when it ends.
**Needs:** S5's export (on main); Nick's merge of the Build 2 spec; S6 bite 1's DESIGN decisions. After that
it can run beside S6–S7.

### S9 — Evaluate, model list, compare  `[ ]`
**Goal:** spec stages 9 and 10 for detectors. Design: DESIGN §Planned for S9; Lead calls D90–D94. One draft PR
per bite, about 300 lines each. Bites 1–2 need only a trained model folder (a fake one in tests).

**Bite 1 — Evaluate job and tiled inference** (D90)
- [ ] the trainer stores `train.eval_scale` (44 px over the training split's median shorter box side) in
      `recipe.json` at the start of a run; models trained before it get it worked out from their dataset version
- [ ] job kind `evaluate`: one model × one test set → `eval/<version>-<test hash>/` (`eval.json`,
      `predictions.jsonl`); frames as items; the eval scale, then tiles of the trained canvas with NMS across
      seams; tests with a fake model

**Bite 2 — The figures and how far they can be trusted** (D91, D92)
- [ ] count error per class and frame (share within ±20 %, the strict share beside it), empty frames with a
      prediction, precision / recall / mAP50 per class, small-object recall, largest object % error; frames and
      objects behind each figure; Wilson for frame shares, frame (or group) bootstrap for the rest; the rank floor;
      synthetic tests, mAP50 checked against pycocotools on a hand-made case

**Bite 3 — Per-frame view**
- [ ] ground truth and predictions drawn, sorted by worst error, in the app's large viewer

**Bite 4 — Model list**
- [ ] `/models`: every model with kind, type, size, dataset version, date, key scores, file size, status from the
      files (DESIGN §Planned for S9), notes in Nick's words, archive (hides, never removes)

**Bite 5 — Compare** (D92, D93)
- [ ] pick models and one test set: one table, one chart per metric, the same frames side by side; differences in
      version, recipe or classes called out; the paired rank rule; "Evaluate on this test set" for a model without
      an eval there; a model that saw a test frame in training flagged

**Bite 6 — RF-DETR Nano as a laptop reference** (D94; time-boxed to one day)
- [ ] a second trainable detector type, laptop only; starting weights only with their licence recorded

**Demo (Nick):** train a second detector with a different size or recipe; the compare page shows old vs new on
the same test frames, and which frames each one gets wrong.
**Needs:** S8.

### S10 — Segmenter and instance segmenter  `[ ]`
**Goal:** the other two kinds go through the same train / evaluate / compare pages.
- [ ] camera models only from architectures that run in S6–S7's table; others are laptop references
- [ ] Lead call in DESIGN before training: how the segmenter treats "unknown" (a class in the MVP spec) and
      classes not fully checked on a frame (`seg_masks/` cannot show them; the flags are in `frame_metrics.csv`)
- [ ] per-pixel segmenter from `seg_masks/`, classes as in S8 with every other pixel as "other";
      cover error in percentage points per class (share within 10 points), IoU per class, largest object % error
- [ ] instance segmenter from the COCO outlines: on the laptop, and for the N6 if `yoloxproto` or RTMDet-Ins-tiny
      runs there in S7 (Nick, build2 spec answer 16: own permissive instance segmenters for the N6); its counts
      and cover scored like the other two
**Demo (Nick):** train a segmenter and an instance segmenter; the compare page shows % cover error per class
and the same frames with each model's masks.
**Needs:** S9; S6–S7's table.

### S11 — Trained models on the cameras  `[ ]`
**Goal:** spec stages 11 and 12 for trained models: what converting costs, and accuracy on the boards.
- [ ] a trained model goes through S6's convert job with calibration frames from the training split; the 8-bit
      model is evaluated on the laptop like any model; the compare page shows what converting cost
- [ ] accuracy run on S7's bench runner: send the test frames, read back raw outputs, score as in S9 / S10; the
      gap to the laptop's 8-bit result is shown (a gap means a runtime bug)
- [ ] pick: one detector and one segmenter per camera marked "picked for the bake-off"; a one-page report
**Demo (Nick):** press "Fit to cameras" on a trained detector and a segmenter; the test session runs the
card on the rig: on-camera scores next to the laptop's. When the card's repeats are done, mark a pick and open the report (the
page refuses a pick from a single run).
**Needs:** S10 (S9 for the detector alone); S7; the rig with the test session. At the end of S11's bench work
the rig is restored to the backup unless Nick says to keep it.

---

## Icebox (captured, not scheduled)

- "Suggest clips" and "Pre-screen with SAM" buttons: left out of S2; decide after S3 (a tiled SAM pass costs about a minute per frame).
- Re-check the kelp prompts (D29) and urchin defaults on more restoration-site footage once the full pipeline
  works and more video is available (Nick, 2026-11-25: only the SSD's videos for now).
- Import CVAT outline fixes back into the project (D42; not asked for in Build 1).
- Later MVP milestones (quality gate, on-camera metrics, bake-off, Iridium packet): after Build 2.
- Build 2: the screen-in-loop bench method on the Pi 5 screen rig (reuse `ADIN_SPI_OpenMV/pi/hil/`); pre-labeling new
  frames with a trained model; a model type per camera when one recipe cannot serve all three.
