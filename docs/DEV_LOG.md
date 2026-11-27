# DEV_LOG.md — Session Log

*Newest entries on top. One entry per working session. Short: what changed,
what broke, what's next.*

---

## Entry template

```
## YYYY-MM-DD — Sprint Sn — <one-line summary>
**Branch:** sprint/n-name
**Done:**  <bullets>
**Broke/surprised us:** <bullets or "nothing">
**Next:** <the single next bite>
```

---

## 2026-11-26 — Build 2 — Spec written with Nick (spec lead)
**Branch:** sprint/6-build2-spec (docs only)
**Done:** `docs/source/build2_spec.md` (train, evaluate, compare, fit to the cameras, bench test, in the same app);
SPEC goal, inventory, hard constraints 1, 6 and new 9, non-goals, Q19–Q24; TRACKER ladder S6–S11. Nick answered seven
questions in the spec lead's chat (instance segmentation laptop first; permissive only; test frames may go to his bench
units; two of his rigs as bench units, he runs bench tests; frames from file first; outside data from his earlier work).
Prior art read: `ADIN_SPI_OpenMV/ml/` (YOLOX on both NPUs, measured), its `pi/hil/` rig and dataset licence records.
Licences and toolchain facts carry "lead" (read at the source by the spec lead) or "helper" (to be re-read by the
builder) in the spec.
**Broke/surprised us:** hard constraint 1 ("images never leave the laptop") ruled out any bench test with real frames
until Nick allowed bench units. ST's own zoo models may run only on ST chips, so they cannot be the shared model.
Ultralytics is AGPL for trained models too. The Pi has no inference code in any of Nick's repos yet.
**Next:** Nick merges the spec PR (spec, goal, scope and hard constraints are his merge); then S6 bite 1.

## 2026-11-26 — Sprint S5 — Fix: add a missing urchin type (D50)
**Branch:** sprint/5-add-urchin-type
- Done: projects made before D39 have no black urchin, so the Review page had no B key (found in Nick's S5 test).
  The Review page now names each missing starting urchin type with one button, "Add black urchin": it appends the
  class to the end of `classes.yaml` (id 10 if unused, else the next new id) with the usual checks; the other classes
  and SAM results stay as they are. `classes.missing_types` / `add_type`, route `POST /p/<name>/labels/add-type`,
  tests `tests/test_missing_type.py`; full suite 296 passed.
- Self-test: a scratch copy of the S5 project with the black urchin line removed showed the note; the button
  appended id 10 at the end, the note went away, and the frame's panel showed "black urchin (B)".
- QE PR 58 nit fixed: a type removed after it had SAM results gets back the id its results hold (a new id would be
  refused by the D6 check); test added, full suite 297 passed.
- Broke: nothing seen. Next: Nick clicks the button on his older project.

## 2026-11-26 — Sprint S5 — Bites 4b, 5b, 6a, 6b (builder, overnight)
**Branch:** sprint/5-fast-tagging (PR 54), 5-add-on-page (PR 55), 5-export-page (PR 56), 5-licences-readme (PR 57)
**Done:** fast urchin tagging on the Review page (Tab zooms to the next untagged urchin, P / R / B tag and move on,
close-up panel, "Tag all remaining as purple"); add a missed object on the page (click or drag a box → a `sam_add`
job; real SAM added a missed urchin, score 0.94); the Export page (Export as a job, checks PASS / FAIL, splits, the
D43 warning, per-class counts); `THIRD_PARTY_LICENSES`, README, the CVAT check (docs only, not tested end to end),
9 of 10 acceptance boxes ticked with evidence. Lead calls D47–D49 logged by the lead (PR 53).
**Broke/surprised us:** the close-up `<img>` took the viewer's absolute-position rule and covered the panel; a
clicked button kept focus so Enter could press it twice; the Export page did not refresh after the job until
`data-reload-on` was added. Stacking PRs on merge commits made rebases slow: prefer one parent per PR.
**Next:** the lead hands Nick the S5 test build; the open box is his `frame_metrics.csv` spot check on 5 frames.

## 2026-11-26 — Sprint S5 — Bites 3a, 3b, 5a (builder, overnight)
**Branch:** sprint/5-review-page (PR 50), 5-review-checks (PR 51), 5-add-object (PR 52)
**Done:** the label Review page (`/p/<name>/labels`): frame list, S2's viewer reused with a click hook, picture
modes, click to select, class picker, X / Enter / Z, collapsible panel; split check, fully checked per class, running
totals over accepted frames; add-a-missed-object backend (SAM 3 click / box on a crop, job kind `sam_add`). Browser
self-tests on the scratch copy with real SAM results; a real-SAM click: 16.9 s per add (D46: no kept worker).
**Broke/surprised us:** the browser pane's viewport (1024x768) is not the screenshot frame (800x600): click by
element reference, not by screenshot coordinates. PRs 47 and 50 both add an as-built bullet at the same place in
DESIGN, so the second to merge needs a rebase. QE PR 44: a deleted mask hid another class (fixed: own run unit only).
**Next:** bite 4b (P / R / B, Tab zoom, close-up, tag all) on the Review page; 5b (click / box on the page); 6.

## 2026-11-25 — Sprint S5 — Bites 1, 2, 4a, 7 (builder, overnight)
**Branch:** sprint/5-review-state (PR 44), 5-export (PR 46), 5-urchin-tagging (PR 47), 5-sam-quality (PR 49)
**Done:** review state in `state.sqlite` with stable mask ids, final masks after review, accept rule, undo (bite 1);
export writers + splits + metrics + checks (bite 2; new dependency contourpy for COCO outlines); black urchin, renames
that keep the id, tagging helpers (bite 4a, backend of bite 4); areas on the whole frame at 2016 px and the anemone
prompt, from a 10-frame real-SAM bench (bite 7, Lead calls proposed). Self-tests on a scratch copy of an S4 project
(16 real 4K frames): export checks PASS, polygon IoU median 0.997; rename re-runs nothing; Preview re-ran areas only.
**Broke/surprised us:** a run unit's key is the class name, so a rename would have orphaned its results
(`follow_renames` re-keys on save). macOS `._*` files on the exFAT SSD tripped the split check. "sea anemone" takes
27 of 94 urchins under rule 1; whole-frame areas miss on 2 of 10 frames. QE: a deleted mask hid another class (fixed).
**Next:** bite 3 (Review page) now that S4's page PRs are merged; then 4b, 5, 6.

## 2026-11-25 — Sprint S4 — Closed (planning lead)
**Branch:** sprint/4-close (docs only)
**Done:** S4 marked `[x]`. Nick passed the checklist at 10:13 PM (lead chat). His feedback from the test went in as
bites 6 (PR 41: Run panel order, Cancel at once, harmless CPU fallback in the log), 6b (PR 42: Resume, progress,
pages in sync, memory-guard message and auto-resume, project page) and 6c (PR 45: "Labelled so far", project row
from disk); Nick re-checks them in the S5 test build. Lead calls D34a, D36a–D36c.
**Broke/surprised us:** the memory guard paused Nick's run while he used the laptop (1.8 GB free); the reason was in
the log only, now on every page, with auto-resume.
**Next:** S5 bites, overnight.

## 2026-11-25 — Sprint S4 — Bite 6c: Labeled so far, project row from disk (builder)
**Branch:** sprint/4-labeled (stacked on sprint/4-feedback-b)
**Done:** Nick's items 9, 9a, 8a: a "Labeled so far" grid on the Labeling page (newest first, live while Run goes
on, legend and large viewer shared with the preview), and the project page's SAM labeling row counted from disk.
Self-tested with real SAM: a new frame appeared first in the grid during Run with no page reload.
**Broke/surprised us:** the large viewer only existed with a preview; it is now always on the page. I first
numbered this "bite 8"; renamed to the lead's 6c.
**Next:** QE on PRs 42 and 45; the lead closes S4 after they merge.

## 2026-11-25 — Sprint S4 — Bite 6b: live job boxes, Resume, memory-guard auto-resume (builder)
**Branch:** sprint/4-feedback-b (stacked on sprint/4-feedback)
**Done:** Nick's items 4–8: one job panel and banner (`_job.html`) on every page, live fragments + `live.js`
instead of reloads, progress bar with a running sign and "done by", Resume next to the state and in a banner,
memory-guard reason on the job and auto-resume (D36c), job status on the project page. Self-tested with real SAM:
paused from the Jobs tab, the Labeling tab followed with no reload; resumed from the banner, it followed back;
a 40 GB limit gave "Paused by the memory guard: 22.6 GB free (limit 40 GB)" on the page.
**Broke/surprised us:** Preview / Run stayed greyed after a pause until reload: buttons now follow the job
(`data-job-lock`). A background tab stops polling and catches up within 2 s when shown.
**Next:** QE on PR 42; then bite 6c (Labelled so far, project row).

## 2026-11-25 — Sprint S5 — Plan (planning lead)
**Branch:** sprint/5-plan (docs only)
**Done:** S4's test checklist passed (Nick, lead chat, 10:13 PM: "I've tested everything in the check list and it
passes"); his feedback items are S4's last bites, re-checked in the S5 test build. S5 planned under CLAUDE.md
rule 7: seven bites in TRACKER S5 (review state, export writers, review page, urchin types and fast tagging, add a
missed object, export page + licences + README, SAM quality), DESIGN §Planned for S5, Lead calls D40–D46 (SPEC Q9,
Q11). Nick saw straight edges on area masks at tile seams on several frames: bite 7 (D45).
**Broke/surprised us:** there is no page yet to see a finished run's frames; S4's "Labelled so far" item covers it.
**Next:** QE on the plan; the S5 builder starts bite 1 tonight.

## 2026-11-25 — Sprint S4 — Bite 6: Nick's test-build feedback (builder)
**Branch:** sprint/4-feedback
**Done:** Run panel reordered (time, finish clock time, then a class | frames-to-run table), Estimate, Preview and
job status lines in the same order; Cancel ends the worker at once, also while pausing (D34a), and a pausing job
says what it waits for; `aten::_assert_async` goes to the job log only (D36b). Self-tested with real SAM on the
scratch folder: Pause, then Cancel while pausing ended the job 3 s after the click; frame 9 was not written.
**Broke/surprised us:** the first try was paused by the memory guard (1.8 GB free) while the test suite ran
alongside, which is the guard working; a quiet second try passed. "pausing — pausing: …" read twice; reworded.
**Next:** QE on the PR, then Nick's S4 re-test.

## 2026-11-25 — Sprint S4 — Bites 1a–5b (builder)
**Branch:** sprint/4-jobctl-core, 4-sam-jobs, 4-extract-jobs, 4-estimate, 4-limits, 4-schedule, 4-awake (PRs 32–37, 39)
**Done:** job control as `app/jobctl/` (one queue, records in `<data_dir>/jobs/`, workers in their own session with
`pause` on stdin, interrupted after a quit, one app per data folder); SAM, extraction and playback copies on it
(`app/jobs.py` removed; Stop became Pause); Jobs page and dashboard "Unfinished job"; Estimate (1 + 5 frames) and
rolling remaining time; limits (device, threads, nice, pause, memory guard on psutil available, measured: D36a
2 GB) and the CPU-fallback warning; Start at / Stop by with Start now and missed; keep awake, per-job summary,
morning summary, log page. Each bite self-tested in the browser, most with real SAM on the scratch data folder.
QE found real races: a quick restart + Resume ran an item twice (fixed: worker PID kept in the row, the queue
waits for it); an Apply while paused was undone on Resume; Resume dropped a future Stop by.
**Broke/surprised us:** my scripted TRACKER edit once matched S2's "Bite 2" and pasted 117 lines (fixed; edits are
now scoped to the S4 section). `vm_stat` "Pages free" reaches 0 GB in ordinary SAM runs; psutil available does not.
A test raced the runner (files ready before the job row said done); fixed. The laptop was on battery during the
keep-awake self-test: the warning showed, and macOS listed the process's idle and system sleep assertions.
**Next:** QE on PRs 37 and 39; then the lead hands Nick the S4 test build (TRACKER Demo).

## 2026-11-25 — Sprint S4 — Plan (planning lead)
**Branch:** sprint/4-plan (docs only)
**Done:** S4 (job control) planned under CLAUDE.md rule 7: five bites in TRACKER S4, DESIGN §Planned for S4, Lead
calls D32–D38 (module `app/jobctl/` with one queue and records in `<data_dir>/jobs/`; worker protocol with `pause`
on stdin and progress from the artifacts; pause / resume / interrupted jobs; estimate 1 + 5 frames; limits and a
measured memory guard; schedule; keep awake). SPEC Q10 points at D36.
**Broke/surprised us:** a hook on this laptop blocks shell commands that name the macOS keep-awake tool: builders
test keep awake through the app, and tests use a fake command (DESIGN §Planned for S4).
**Next:** the S4 builder starts bite 1a.

## 2026-11-25 — Sprint S3 — Demo passed
**Branch:** sprint/3-demo-passed (docs only)
**Done:** Nick ran the S3 test build on `main` at `49e1db8` and wrote "S3 passes" in the planning lead's chat
(5:03 PM). Bites 1–6 merged: PRs 20, 21, 22, 24, 27, 28; docs PRs 19, 23, 25, 26. Lead calls D22–D29, D31; Nick's
class form D30 replaced D25. S3 marked `[x]`; bites 1–3 ticked with their PRs.
**Broke/surprised us:** kelp prompts are weak (n = 6 frames; shoots come in pieces): re-check on more footage
later (icebox). Whole-frame SAM on a 5.3K frame peaked at 26.7 GB Apple GPU memory (input for S4's memory guard).
Two builders were briefly started for bites 5–6; the duplicate was archived before it changed anything.
**Next:** S4 (job control), planned by the lead.

## 2026-11-25 — Sprint S3 — Bite 6: class editor form
**Branch:** sprint/3-class-form (from sprint/3-viewer, bite 5)
**Done:** the YAML text box is replaced by a form: rows with name, kind, prompt chips, group and "pixels
count as" drop-downs, ↑ ↓, Remove, "+ Add class" with the next free id, and the SAM settings as labelled
number fields. Every change is checked on the server with the same checks as classes.yaml; Save is off
while there is a problem. A fuzz test (1,200 random form inputs) shows no input saves a file that fails
the checks, and a refused save leaves the file untouched. 168 tests.
**Broke/surprised us:** leaving a text box redrew the table and dropped the click on the next box; the
redraw now waits for the focus to move and gives it back.
**Next:** QE on the bite 5 and 6 PRs.

## 2026-11-25 — Sprint S3 — Bite 5: large viewer and class legend on the Labeling page
**Branch:** sprint/3-viewer (PR 27; built on bite 4's branch, `main` merged in after PR 24's squash)
**Done:** clicking a preview frame opens S2's viewer on the page (arrows, Esc, zoom, pan) with the masks;
a class legend (toggle, "only", final class map) drives the tiles and the viewer together; objects boxed,
areas not, and the legend says so. Masks are decoded inside their box only and cached: on a synthetic 5.3K
frame with 150 masks a toggle went from 1.9 s to 0.01–0.05 s (0.5 s the first time). 161 tests.
**Broke/surprised us:** a class toggle in the viewer reset the zoom; `viewerShow(i, true)` now keeps it.
**Next:** QE on the bite 5 PR; then bite 6 (class editor form).

## 2026-11-25 — Sprint S3 — Bite 4: Run, Stop, re-run only what changed, overlap rules
**Branch:** sprint/3-run (PR 24)
**Done:** Run on every kept frame with Stop and carry-on; the page lists the classes that will run and those
unchanged; `app/sam/overlap.py` (D22) with a "final class map" toggle. Self-test with real SAM on 8 4K frames:
Stop at frame 2, Run carried on; a new class ran alone, and the other classes' results stayed unchanged.
**Broke/surprised us:** the first estimate (preview s/frame × the share of units) said 1 min for one new class;
it took 3 min 43 s, because reading, tiling and image encoding cost about 21 s per frame however few units run.
The estimate is now a fixed cost per frame plus a per-unit cost.
QE fixes (PR 24 FAIL at `77b11bd`): class ids capped at 254 (the class map is one byte, 255 = untagged);
a frame with an unreadable `sam_raw` file is skipped and named instead of stopping Run; Stop only shows during a
Run; D31 in `overlap.py` (a reviewed mask beats an unreviewed one at IoU > 0.5).
Nick's feedback on the Labeling page (viewer, mask legend, class editor) is captured separately.
**Next:** QE on PR 24; then Nick's feedback items.

## 2026-11-25 — Sprint S3 — Bite 3 bench: kelp prompts, D10 re-check, minimum mask area
**Branch:** sprint/3-preview (main merged in after PR 21's squash, `c24dba4`)
**Done:** kelp prompts picked from the overlays of 6 restoration frames: kelp shoot "young kelp", kelp area
"kelp" ("kelp stipe" found nothing; "kelp forest" covered open water). D10 re-checked on 9 urchin frames from
5 videos (3 sites): tiles keep 3–4× more urchins on dense 4K frames, the same on close 5.3K frames, at 4–8× the
time; D10 holds. D24: the smallest of 580 urchin masks is 948 px and the small ones are real, so
`min_mask_area_px` 500 (a floor for specks). Numbers in DESIGN §Bench results.
**Broke/surprised us:** whole-frame SAM on 5312 px frames peaked at 26.7 GB Apple GPU memory (tiles 16.3 GB).
Three Wreck videos are 1280×960, so they are never tiled.
**Next:** draft PR for bite 3 to QE; then bite 4 (Run, re-run, overlap rules).

## 2026-11-25 — Sprint S3 — Bites 1–2 built, bite 3 preview in progress; builder rotates
**Branch:** sprint/3-classes (PR 20, merged `373521d`), sprint/3-sam-runner (PR 21), sprint/3-preview (no PR yet)
**Done:** bite 1: `classes.yaml` with checks naming the line, Labeling page with edit / Save / Reload, upright
EXIF helper (review thumbnails fixed). Bite 2: SAM worker process, `sam_raw/<frame file>.json`, tile-edge
duplicates (D23, cut pieces unioned after QE), area masks one per prompt, re-run by settings key. Bite 3 code:
Preview on 3 frames with toggles, counts, s per frame and peak memory; ran with real SAM in the browser
(65 s per 4K frame, 12.8 GB Apple GPU). 132 tests.
**Broke/surprised us:** the bench frames from 3 videos showed urchins on 2 frames only; "kelp stipe" found
nothing on 6 restoration frames. macOS `._` files appear in `sam_raw/` on the exFAT SSD.
**Next:** bite 3's kelp prompts, D10 re-check on more urchin frames, D24 minimum area, PR (handoff note in the notes folder).

## 2026-11-25 — Sprint S3 — Plan
**Branch:** sprint/3-plan (docs only)
**Done:** planning lead wrote the S3 plan (CLAUDE.md rule 7): four bites in TRACKER (classes, SAM runner and
results files, prompt preview, run + re-run + overlap rules), DESIGN §Planned for S3, Lead calls D22–D27
(SPEC Q2 overlap rules and kelp shoot; Q10 SAM defaults; classes edited in the browser; preview frames;
EXIF orientation of stills).
**Broke/surprised us:** nothing.
**Next:** QE reviews the plan; then the S3 builder starts bite 1.

## 2026-11-25 — Sprint S2 — Demo passed
**Branch:** sprint/2-demo-passed (docs only)
**Done:** Nick ran the S2 test build and wrote "the demo passes" in the builder chat (12:09 PM).
Every S2 PR is merged (last code PR: PR 15, filter rows, `4fad379`; then docs PR 17, `106f867`). S2 marked `[x]` in TRACKER.
**Broke/surprised us:** nothing new.
**Next:** S3 (SAM on single frames), planned by the lead.

## 2026-11-25 — Sprint S2 — Bites 4 (review frames) and 5 (stills + resume) merged
**Branch:** sprint/2-frames-4 (PR 14), sprint/2-frames-5 (PR 16), sprint/2-frames-45-done (docs only)
**Done:** Bite 4: `app/stages/review.py`; grid of 640 px copies with a 1–8 per-row slider, arrows,
`D` (to `rejected/`), `K`, `Z`; full-screen viewer shared with Extract (`viewer.js`); "Show dropped"
with `K` = `kept_hand` (D20); Skip and Done. QE Lows fixed: undo keeps a step whose frame Extract
changed (the hand mark survives a plan change); preview and status counts use the plan. Bite 5:
still-image folders copied byte for byte into `frames/`, counted apart, skipping clips, extraction
and filters (D21, Lead call); a truncated JPEG is counted unreadable (QE Medium); Extract shows how
far a stopped run got and carries on. 94 tests. S2 Demo run end to end in the builder's browser
on the bite 5 head (notes folder, `s2-demo-run-2026-11-25.md`).
**Broke/surprised us:** force-push is blocked in builder sessions, so branches take `main` by
merge, not rebase; PR 16 was stacked on PR 14 and retargeted after the squash merge.
S2 Demo re-run on `main` at `4fad379` (after PR 15): 37 kept + 3 blurry + 0 duplicate = 40 = live
count; D / Z / D; restart kept all four stages and the frame in `rejected/`; 6 stills in review.
**Next:** Nick's S2 test build; the checklist is `s2-test-build-checklist.md` in the notes folder.

## 2026-11-25 — Sprint S2 — Filter rows simplified (Nick's feedback on 3b)
**Branch:** sprint/2-filter-rows (from main `87310c0`)
**Done:** Nick found the filter examples too much and their labels confusing ("the sharpest
frames dropped"). Now one row per filter showing every frame it would drop, scrolling sideways; click
opens the viewer. Order (Nick's second look): most blurry and least unique (fewest bits
differing from the last kept frame) on the left. The blur row follows the setting at which each
frame drops, so a stronger blur only adds on the right (QE); the duplicate row can still gain
frames mid-row.
After PR 14 (bite 4) merged: main merged in, `kept_hand` kept (never ranked by blur, still a
duplicate anchor), and the rows load the 640 px `/framethumbs/` copies; the viewer still opens
the full-size frame. 89 tests. The kept-side and paired examples are gone. Browser
check on the scratch project at 30 % / 20: rows of 12 and 14 in order (sharpness 4 → 141,
11 → 19 bits). 82 tests.
**Broke/surprised us:** Nick's app on port 8000 was an old copy from the previous builder's
folder, so he saw no filters at first.
**Next:** QE reviews the draft PR; the lead merges.

## 2026-11-25 — Sprint S2 — Bite 3b merged; builder rotates
**Branch:** sprint/2-frames-3b-done (docs only)
**Done:** PR 9 (bite 3b) merged by the lead as `1284232`, matching QE's PASS on `5a72243`, under
the new working rules (PR 10, `36b499f`). TRACKER 3b items marked done.
**Broke/surprised us:** Nick did not run the 3b manual test before the merge; under the new rules
it moves into the S2 test build (the four filter steps in PR 9's description).
**Next:** bite 4 (review frames) by the next builder session; handoff note in the session notes.

## 2026-11-25 — Sprint S2 — Bite 3a merged; bite 3b: blur and duplicate filters
**Branch:** sprint/2-frames-3b (from main `f9f14b2`, after PR 7 merged)
**Done:** PR 7 (bite 3a) merged as `f9f14b2` after QE PASS on `5a3d96d` and Nick's word. Bite 3b
plan approved by Nick. `extract.py`: a pure filter rule (per video, blur first, then duplicates
against the last kept frame), a read-only preview with examples at each cut-off, and Apply,
which moves files between `frames/`, `dropped/blurry/` and `dropped/duplicate/` (never erasing)
and repairs a file left in the wrong folder by a stop. Extract applies the saved values at its
end and never decodes a frame already in one of those folders (QE PR 7 watch item). Extract
page: two sliders with live counts, example frames (click opens the viewer), Apply. 8 new tests
(82 in total). Browser check on the scratch copy of 44 real frames: 30 % / 20 bits gave
18 + 12 + 14 = 44 and the folders held 18 / 12 / 14 files; Extract again decoded nothing;
0 % / 0 brought all 44 back.
**Broke/surprised us:** with the default duplicate value an older test's synthetic video lost a
frame as a duplicate (its frames look alike), so that test turns the filters off. Example
frames are full-size 4K files and take a moment to appear.
QE PASS-with-nits on `845ec27`: rebased on main, decision renumbered D19 (D18 is the urchin type from PR 8), a non-integer duplicate value now gets 400.
**Next:** QE reviews the draft PR; Nick runs the 3b manual test; then bite 4 (review frames).

## 2026-11-25 — Sprint S2 — Bite 3a: QE findings fixed
**Branch:** sprint/2-frames-3 (PR 7, still a draft)
**Done:** QE PASS-with-nits at c182fa2 had three findings. (1) Frames from earlier settings stayed `kept`
  (12 kept for a plan of 3): they now move to `dropped/unplanned/` and come back if asked for again (D17a).
  (2) Rotated videos would fail every frame: decoding goes through a PPM pipe, so frames come out upright.
  (3) TRACKER 3a items marked [x]. Two tests added; 72 pass.
  Nick asked for an in-page frame viewer (like iPhoto): click a frame to see it large, arrows or < > move
  through every kept frame, Esc or Close leaves. Then full screen (browser Fullscreen API), no frame
  around the image; click zooms to actual pixels at that point, scroll / pinch zooms, drag pans, 0 fits.
  Nick did not see the changes: the browser kept old saved copies of the scripts. Script and style
  links now carry a content hash (`?v=`) and /static answers `Cache-Control: no-cache`.
  QE PASS-with-nits at 730b58e; its Low finding fixed: a stop between moving a frame and updating its
  row is repaired on the next run (test added, 74 pass). The 3b watch item is in TRACKER.
**Broke:** nothing known.
**Next:** QE verdict on the new head (new QE session not running yet); Nick's manual test; then bite 3b.

## 2026-11-24 — Sprint S2 — Bite 3a: sampling plan, extraction, counts
**Branch:** sprint/2-frames-3
**Done:** Nick said to keep working in this session (instead of rotating). `app/stages/extract.py`:
sample times inside clips (whole video if none), cap spread evenly over all samples; frames
decoded from the original with ffmpeg and saved by Pillow at JPEG quality 95 as
`<video-stem>_<path-code>_<ms>.jpg` (spaces in the stem become _); blur score and dHash stored per
frame with its video and timestamp; temp name first, one database row per frame, so a re-run only
makes missing frames. Extract page: settings, live count while typing, background job with
progress, counts, the first 12 frames. 7 new tests (70 in total). Real run on 2 SSD videos
(scratch project): 44 frames as planned, 3840 px wide, 46 MB, 20 s; blur scores 4.5–584.
**Broke/surprised us:** nothing.
**Next:** QE reviews the draft PR; Nick runs the 3a manual test; then bite 3b (filters).

## 2026-11-24 — Sprint S2 — Bite 2 merged; bite 3 planned (session rotates)
**Branch:** sprint/2-frames-3 (from main `c730e20`)
**Done:** PR 6 (bite 2b) merged as `c730e20` after QE PASS on `5e52833` and Nick's "ship it".
Bite 3 plan approved by Nick: split 3a / 3b, blur cut-off per video, dropped frames moved to
`dropped/` folders (DESIGN D17). Measured on two SSD videos (scratch output only): full-size frame
0.5 s (4K) / 0.8 s (5.3K) by seek; JPEG q95 save 0.02–0.04 s, 1.5 / 2.0 MB; blur + dHash ~0.03 s.
**Broke/surprised us:** sharpness scores differ ~5x between videos (why D17 is per video). A running
app copy serves new templates with old code after a branch switch (500s in Nick's log); restart fixes it.
**Next:** bite 3a in a fresh builder session (TEAM_RULES 9: this one covered bites 1–2).

## 2026-11-24 — Sprint S2 — Bite 2b: clip marks (I / O), clips.json, counts
**Branch:** sprint/2-frames-2b (from main `9b296cd`, after PR 5 merged)
**Done:** plan approved by Nick. Timeline under the player with clip bands and a playhead
(click to jump); keys I / O / Space / arrows (Shift = 5 s), ignored while typing a note; clip list
with note, Go to, Start = now, End = now, Delete; every change saved at once to `clips.json`
(atomic). Server checks each video's clips: order, overlap, at least 0.5 s, inside the video.
Counts per video and in total at one frame every 2 s (whole video when unmarked); Pick clips row
on the project page shows "N clips, M:SS marked". 4 new tests (61 in total). Browser check with
real key presses on a real video: I at 0:05, O at 0:15 made the clip; an overlapping clip was
refused with its message; a note survived a reload; Delete returned the video to "whole video".
From Nick's manual test: clearer refusal messages and button names (Set start here / Set end
here, hover hints); a clip's band and its row light up together when either is clicked; each row
shows the clip's first frame (made from the copy at that time, cached in `clipthumbs/`), which
changes when the start is trimmed. 63 tests.
**Broke/surprised us:** Nick's 2a test found an empty strip: macOS writes `._` shadow files next
to every file on the SSD (fixed in PR 5).
**Next:** QE reviews the draft PR; Nick runs the 2b manual test; then bite 3 (extract frames).

## 2026-11-24 — Sprint S2 — Bite 2a: playback copies and player
**Branch:** sprint/2-frames-2a (from main `4d11e30`)
**Done:** bite 2 plan approved by Nick: split 2a (player) / 2b (marks), copies 960 px wide.
`app/jobs.py`: one background worker, jobs queued and run one at a time with percent and time
left. `app/stages/clips.py`: 960 px H.264 playback copy (Mac's built-in encoder; a GoPro .LRV
is repackaged instead) and a strip of one thumbnail every 5 s made from the copy, both written to
temporary names first. Pick clips page: video list with progress, the player, the strip (click to
jump). 6 new tests (55 in total). Browser check on 2 real 4K videos: the copy was made in the
background, the page reloaded when ready, clicking the 0:20 thumbnail jumped the player to 20 s.
**Broke/surprised us:** measured before coding: decoding the originals with the Mac's video chip
was slower on 5.3K and failed on 4K, so originals are decoded in software. GoPro files carry a
data track that has to be left out of the copy.
**Next:** QE reviews the draft PR; Nick runs the 2a manual test; then bite 2b (I / O marks).

## 2026-11-24 — Sprint S2 — Bite 1b: ingest page
**Branch:** sprint/2-frames-1b (from main `b00be85`, after PR 3 merged)
**Done:** plan approved by Nick. `app/stages/ingest.py`: find .MP4 / .MOV under a folder and its
subfolders, skip `._` files, .LRV / .THM and hidden folders; ffprobe details plus a one-frame
decode check; GoPro chapters grouped; same name + size counted once ("also in"). Ingest page with
a folder browser (starts at `video_roots`, or a pasted path), tick boxes, add / remove; the
project's Ingest row shows "done · N videos (M recordings)". 12 new tests (43 in total), one
proving source files are byte-identical with the same modified time afterwards.
On the SSD: 29 videos listed, 17 with "also in", none failing the decode check. The first page
load takes ~6.6 s (ffprobe + decode of 29 videos); later loads ~0.1 s (cached while the app runs).
Nick's manual test (2 videos added) worked; he asked for a Finder-style thumbnail per video:
added (one frame 10 % in, 240 px, cached in `<data_dir>/thumbs/`); the first load that builds all
29 took ~16 s, later loads reuse them.
**Broke/surprised us:** of two copies, the one in the folder that sorts first is the one listed.
`uv run app` right after stopping a copy said "port in use" (closing connections); fixed.
**Next:** Nick runs the 1b manual test; QE reviews the draft PR; then bite 2 (clip picker).

## 2026-11-24 — Sprint S2 — Bite 1a: app skeleton (start, settings errors, projects)
**Branch:** sprint/2-frames
**Done:** bite 1 plan approved by Nick, split into 1a (skeleton) and 1b (ingest); same-name,
same-size videos will count as one. `uv run app` serves on 127.0.0.1:8000 and opens the dashboard.
Settings problems show as a red box in the browser and the settings file is re-read on every
page load. Projects: create / open / list under `<data_dir>/projects/`, with `project.json` and
`state.sqlite`; the project page lists the six stages. 14 new tests (31 in total) pass.
**Broke/surprised us:** FastAPI form posts need `python-multipart` (Apache-2.0), which the
plan did not list. The SSD's video folder holds 46 videos but only 29 distinct ones: 17 are
copies in a second folder. No GoPro-named files there, so chapter grouping can only be tested on
made-up names.
**Next:** Nick runs the 1a manual test; QE reviews the draft PR; then bite 1b (ingest).

## 2026-11-24 — Sprint S2 — Plan written (planning lead)
**Branch:** sprint/2-frames
**Done:** S1 merged to main as `a362495` (QE PASS on `db29464`; Nick ran the demo). S2 plan
approved by Nick: five bites in TRACKER, planned design in DESIGN, D12. Kelp and kelp shoot are
starting classes again (D11). SPEC Q1, Q6, Q13 and part of Q10 answered.
**Broke/surprised us:** the S1 demo video had almost no urchins but did have kelp.
**Next:** S2 bite 1, plan nibble — Nick gate.

## 2026-11-24 — Sprint S1 — SAM test runner built; Apple GPU works with run-time fixes
**Branch:** sprint/1-sam-test
**Done:** plan approved by Nick. `pyproject.toml` (uv, Python 3.12, torch 2.10.0, Meta's sam3
pinned at `31cb7e6`). `app/sam_test/`: frame pull, one worker process per attempt, report page
with overlays embedded, click and box test. 9 unit tests on synthetic images (13 by the end of S1). Smoke test with
untrained weights: text, click and box prompts run on the Apple GPU (~2.3 s per 1008 px image).
Dry run on a real video: frames extracted; report shows the "download the weights" error.
**Broke/surprised us:** torch 2.15.0 aborts on the Apple GPU on macOS 14.6 → pinned 2.10.0 (D7).
Meta's image code needs six run-time fixes off-CUDA plus PyTorch's CPU fallback (D8). The bite
is ~630 lines of app code, about twice the plan's ~340, because of the fixes and the worker.
Later the same day: Nick's first real run found nothing — my bfloat16 replacement gave NaN
scores on the Apple GPU (fixed, CPU and GPU now agree). Added `--tile` and `--cutoff`, one folder
per setting combination and an all-runs index page. SAM 3.1 at 504 px tiles: 233 s per frame,
finds urchins only in patches and calls many urchins "sea anemone" (DESIGN, Bench results).
The Hugging Face `LICENSE` for `facebook/sam3.1` matches the GitHub one word for word.
Then (Nick approved the download): SAM 3 gives the same text results as 3.1, but its click
model works (clean urchin outline from one click). The prompt word "urchin" finds far more than
"sea urchin" (DESIGN, Bench results). Added `--prompts`. Licence of `facebook/sam3` = GitHub text.
Nick chose SAM 3 + "urchin" as defaults and to close S1 (D10). Smaller crops do not help
mid-range urchins (504 → 4, 336 → 2, 252 → 0 kept on one hazy patch).
**Next:** Nick runs the S1 demo; QE reviews the draft PR; then S2 plan.

## 2026-11-24 — Sprint S0 — Spec review edits, QE fixes, SAM test sprint added
**Branch:** sprint/0-scaffold
**Done:** Nick ran the S0 demo (3 checks passed). QE gave PASS-with-nits on 51dabf1; its 12
findings are fixed or answered. Ten spec edits from Nick's review written to
`docs/source/mvp_spec.md`. New sprint S1 "SAM test on the laptop" added; later sprints moved
down one. `uv` installed; Nick logged in to Hugging Face and confirmed access to `facebook/sam3.1`.
**Broke/surprised us:** `brew install hf uv` started building `llvm` and `rust` from source on
macOS 14 — stopped; use the standalone `uv` installer and `uvx hf`. The SAM 3 readme lists a
CUDA GPU as a prerequisite and does not mention Apple chips.
**Next:** S1, plan nibble — Nick gate.

## 2026-11-24 — Pre-work — Project scaffolded

**Branch:** n/a
**Done:** docs created from `template_LLM_project`; SPEC and sprint ladder filled from
`docs/source/mvp_spec.md`; `.gitignore` and `config.example.toml` added. Planning session, no app code.
**Broke/surprised us:** nothing yet.
**Next:** Nick runs the S0 demo; then S1 plan nibble — Nick gate.
