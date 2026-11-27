# SPEC.md — urchin_detector_pipeline

*What Nick wants. Stable reference — agents skim this. Changes to the goal, scope or hard constraints
need Nick's approval and his own merge; recording his answers and verified facts does not (CLAUDE.md, rule 3).*
*Last updated: 2026-11-26*

## Goal

A local labeling tool, run in the browser on Nick's laptop, that turns a few GoPro videos
into reviewed training data for two on-camera models: COCO boxes for a detector and
per-pixel class masks for a segmenter. Done when Nick can start the app with one command,
walk six stages in the UI (ingest → pick clips → extract frames → review frames → SAM
labeling → review + export) on his ReefCheck videos (or on folders of still images), and the export passes the acceptance
criteria in `docs/source/mvp_spec.md`.

**Build 2 (added 2026-11-26; source `docs/source/build2_spec.md`):** in the same app, go from that
export to models proven on the bench cameras. Done when Nick can, in the browser, train a detector
and a segmenter on a Build 1 export, compare them with older models and other model types, convert
both for the OpenMV N6 and the Raspberry Pi Zero 2W (the AE3 as a one-day stretch), run them on the
bench units, and read one comparison page that says which model to put on which camera and why.
An instance segmenter can be trained and compared on the laptop; it goes on a camera only if it
converts as easily as the other two.

## Background

- Full spec: `docs/source/mvp_spec.md` ("Urchin Monitoring MVP — Spec", Nick, 2026-11-24).
  That file is the source; this file is the condensed contract. If they disagree, stop and
  ask Nick.
- MVP purpose: choose the go-to-market camera (OpenMV N6 vs Raspberry Pi) from a one-day
  dawn-to-dusk bake-off measuring urchins, sea stars and kelp. This repo is Build 1 (the
  labeling tool) and Build 2 (train, compare, fit to the cameras, bench test: MVP milestones 2–3).
- Prior art for Build 2, read 2026-11-26, to be re-read by the builder before reuse: the local repo
  `ADIN_SPI_OpenMV/ml/` (YOLOX-Nano / -Tiny trainer, 8-bit export, compile scripts for both NPUs,
  on-board decode; results in `ml/yolox_urchin/STAGE1.md` and `ml/compile_gate_report.md`), its
  `pi/hil/` screen-in-loop rig, and its dataset licence records (`ml/urchin_data/`,
  `docs/urchin_datasets.md`); the data and earlier runs are in `~/nereus_ml/` (outside git).
- Prior art to inspect before writing code (S1 plan nibble), in the local repo
  `ADIN_SPI_OpenMV/ml/`:
  - `fomo/label_gui.py` + `test_label_gui.py`: a stdlib-only browser GUI that reviews
    **boxes** in `labels.jsonl` (source: file header, read 2026-11-24). No masks.
  - Reported by a read-only review on 2026-11-24, **to be re-checked in S1 before reuse**:
    path-safety and atomic-save helpers and canvas/keyboard code in `label_gui.py`; COCO
    writers in `rfdetr_gate/prep_coco.py` and `labeler_eval/hil_gt.py`; frame sampling from
    clips in `yolox_urchin/hil_stills.py`. The same review found no blur filter,
    near-duplicate filter, review grid, mask output or SAM use there.

## Inventory / environment

| Item | Qty | Role |
|---|---|---|
| MacBook, Apple M1 Max, 32 GB | 1 | runs the whole tool (Nick, chat 2026-11-24) |
| External SSD `/Volumes/Nick_R_SSD/NereusData` | 1 | source videos (read-only) and project data |
| ReefCheck videos under `nereus-camera-test-rig/2026-10-15_reefcheck` (3) and `2026-10-18_reefcheck_unpack` (29) | 32 files | labeling input (file count 2026-11-24; the two folders may overlap) |
| SAM 3.1 (`facebook/sam3.1`), fallback `facebook/sam3` | 1 | labeling model, laptop only |
| Raspberry Pi: 12 MP IMX708, 2 MP STARVIS IMX462 | — | bake-off cameras (later milestones; from the full spec) |
| OpenMV N6: 5 MP rolling shutter, 1 MP global shutter | — | bake-off cameras (later milestones; from the full spec) |
| OpenMV AE3: 1 MP global shutter | — | stretch goal (later milestones; from the full spec) |
| Bench unit: a Pi Zero 2W rig with the IMX708, an N6 and the AE3 (name in `config.local.toml`) | 1 | Build 2 bench tests; owner Nick (chat 2026-11-26; make-up from the rig repo's SPEC, not re-checked on the unit) |
| Bench unit: a Pi 5 with the screen-in-loop rig (name in `config.local.toml`) | 1 | Build 2 bench tests, optional screen method; owner Nick (chat 2026-11-26) |
| Public urchin datasets and earlier model runs in `~/nereus_ml/` | — | extra detector training data and starting weights, each with its recorded licence (Nick, chat 2026-11-26) |

## Confirmed facts (verified, with sources)

- Two sampled ReefCheck videos are HEVC, 3840×2160, 29.97 fps, 68 s and 182 s long
  (ffprobe, 2026-11-24). Other videos not probed yet.
- Video names are ReefCheck-style (`<site>-<date>-<time>-<diver>-<tag>.MP4`), not
  GoPro `GX01xxxx`; the SSD also holds macOS `._*.MP4` shadow files that are not videos
  (file listing + ffprobe error, 2026-11-24).
- The laptop has ffmpeg / ffprobe and Python 3.13.5 (checked 2026-11-24).
- `uv` 0.13.10 is installed at `~/.local/bin` from the standalone installer
  (`uv --version`, 2026-11-24). The Hugging Face tool is run as `uvx hf ...`; it is not
  installed separately.
- Do not install tools with Homebrew on this laptop: on macOS 14 Homebrew builds from
  source (it tried to build `llvm` and `rust` for `hf` and `uv`; Nick's terminal, 2026-11-24).
- Nick is logged in to Hugging Face and has access to the gated repo `facebook/sam3.1`:
  `uvx hf download facebook/sam3.1 --dry-run` listed 12 files, 3.5 GB in total, the
  checkpoint being `sam3.1_multiplex.pt` (Nick's terminal output, 2026-11-24).
- Coding agents may view real frames (Nick, chat 2026-11-24).

## Safety / hard constraints (non-negotiable)

1. **Images never leave the laptop.** No cloud APIs and no uploads. Exceptions: Claude
   coding sessions may view frames (Nick, 2026-11-24); Build 2's bench test may send test frames
   to Nick's own bench units, over USB, the home network or Tailscale (Nick, 2026-11-26).
2. **No tokens or secrets in git.** Hugging Face access only through `hf auth login`; the
   app never stores the token.
3. **Videos, frames, datasets and model weights stay out of git**, in the data folder named
   in `config.local.toml` (on the SSD). Source code lives in git.
4. **Source videos are read-only.** Never modify, move or delete them.
5. **Delete means move to `rejected/`**, never erase.
6. **Licenses:** on-camera models permissive only (Apache / MIT; BSD-3-Clause added with Build 2). The SAM License is
   acceptable for the labeling tool only. Build 2 (Nick, 2026-11-26): the same rule for every model
   the app can train, laptop-only ones included (no Ultralytics YOLO, which is AGPL-3.0); a model
   type is offered only when the licences of its code, its starting weights and any outside data
   are recorded in `THIRD_PARTY_LICENSES`.
7. **Every stage has a browser UI.** The terminal is used only to start the app. Basic and
   functional, not flashy (Nick, chat 2026-11-24).
8. **The repo is public** (Nick, chat 2026-11-24). No real frames, video stills, people's
   names or tokens in git, PRs, issues or test fixtures. Tests use synthetic images.
9. **Hardware (Build 2).** The app talks only to the two bench units Nick named (SPEC inventory;
   names and addresses in `config.local.toml`), never to a field unit. Nick owns them and runs every bench test himself from the
   app; builder and QE sessions never touch hardware and test against a fake board. Every bench run
   has a test card first. The app never flashes firmware; it loads model files only, backs up what
   it replaces and can restore it. No unit names, addresses or serial numbers in git.

## Success criteria by sprint

See TRACKER.md — every sprint ends with a live demo Nick can run.

## Non-goals (this project)

- Camera firmware, the quality gate, on-camera metrics, the hourly Iridium packet, the field
  bake-off (MVP milestones 4–5). Training and quantizing are Build 2 (2026-11-26).
- Cloud training; automatic search over model settings; pre-labeling new frames with a trained model.
- Hand-editing mask outlines; colour correction of frames.
- Panoptic export; tracking individual animals; kelp growth rates in cm/day.
- Multi-user, cloud, login.

## Open questions (flag, don't guess)

- Q2. Overlap rules for area-vs-area (kelp vs rock) and for several prompts of one class; and
  which segmenter class a kelp shoot's pixels get (kelp is the obvious candidate; not decided).
  **Lead call D22 (2026-11-25):** areas listed first in `classes.yaml` win (kelp over rock); prompts of one
  area class are unioned; kelp-shoot pixels count as kelp. Nick can reverse it.
- Q9. CVAT export format for fine outline fixes, and whether fixes are imported back.
  **Lead call D42 (2026-11-25):** the COCO file with polygon outlines is the CVAT route; no import back in
  Build 1. Nick can reverse it.
- Q10. Defaults still open: minimum mask area, inference resolution, memory guard, and the
  tuned blur and duplicate values (S2 starts from the guesses in DESIGN D12).
  **SAM part, Lead call D24 (2026-11-25):** confidence 0.4, full resolution in 1008 px tiles; minimum mask
  area measured in S3's preview; memory guard moves to S4. Nick can reverse it.
  **Memory guard, Lead call D36 (2026-11-25):** pause a job when free memory falls below a threshold that
  S4 bite 4 measures; the number is logged in D36. Nick can reverse it.
- Q11. "Largest object %" by mask or by box; the id for unknown; which frames count as
  "accepted" for export.
  **Lead call D41 (2026-11-25):** largest object % by mask; unknown = id 0; only accepted frames are exported.
  Nick can reverse it.
- Q12. Does SAM 3.1 run on the M1 Max? **Answered in S1 (2026-11-24): yes, SAM 3.1 and SAM 3
  both run on the Apple GPU with real weights (outcome at the end of this item).** Facts,
  from `facebookresearch/sam3` @ `31cb7e6` (read 2026-11-24): the readme lists Python 3.12+, PyTorch 2.7+ and a CUDA 12.6+ GPU and installs torch 2.10.0;
  the code never mentions MPS. Its image path needs run-time fixes to build off-CUDA (see
  DESIGN D8). With those, PyTorch 2.10.0 and CPU fallback on, a text, click and box prompt ran
  end to end on the Apple GPU with *untrained* weights, about 2.3 s per 1008 px image (smoke
  test, 2026-11-24). PyTorch 2.15.0 aborts on the Apple GPU on macOS 14.6 (D7). The 3.1 release
  notes (`RELEASE_SAM3p1.md`) document video use only; the image builder loads `sam3.pt` by
  default. S1 answer: `sam3.1_multiplex.pt` loads into the image model's detector (only an unused
  neck level is missing) but supplies none of the 309 click/box weights. Licence:
  the repo's `LICENSE` is the "SAM License", last updated 19 Nov 2025: royalty-free, allows
  commercial use and changes; copies passed on must carry the licence; no ITAR, military,
  weapons, nuclear or espionage use. The `LICENSE` in `facebook/sam3.1` on Hugging Face is
  word-for-word the same text, and so is the one in `facebook/sam3` (compared 2026-11-24).
  Results of the S1 runs: DESIGN, Bench / test results.
  S1 outcome (2026-11-24): SAM 3 and SAM 3.1 both run on the Apple GPU (torch 2.10.0, run-time
  fixes, CPU fallback on); Nick chose SAM 3 with the prompt "urchin" as the S1 defaults (D10).
  Open for S3: mid-range urchins in hazy water are mostly missed at any tile size.

Answered (Nick, chat 2026-11-24):
- Repo name stays `urchin_detector_pipeline`; the repo is public.
- Project data lives on the SSD; source code in git (DESIGN D1).
- Laptop is an Apple M1 Max, 32 GB. Agents may view real frames.
- Q1. Kelp shoot (object) and kelp (area) are starting classes again (D11, replaces D5).
- Q3. The live count is before the filters; kept + dropped add up to it.
- Q4. Frame names carry a short code from the video's path; the chapter rule is GoPro-only.
- Q5. No limit on the number of videos; splits go by site and dive date.
- Q7. On a prompt or setting change, re-run that class only; keep reviewed masks, marked as older.
- Q8. A "Keep laptop awake" toggle in the UI plus a plugged-in / lid-open reminder.
- Q9 (part). Missed objects are added by a click or a box in label review.
- Q14. Classes start small and grow toward the Reef Check California manual's species list
  (https://www.reefcheck.org/wp-content/uploads/2021/03/RCCA_Manual_10th_Edition_web.pdf);
  Nick wants to match that standard later (D6). No third
  urchin class now (Nick, chat 2026-11-24). Unverified background: a helper agent's web
  review on 2026-11-24 reported no named third type and that the manual describes red
  urchins as "purple to red to almost black"; nobody has re-read the manual for this yet,
  so re-read it before the class list is extended. Crowned urchin waits for
  footage that clearly shows it. One broad "sea star" class plus "sunflower sea star".
- Q6. "Suggest clips" and "Pre-screen with SAM" are left out of S2; decide after S3 (D12).
- Q10 (part). Blur and duplicate filters start from adjustable guesses (D12).
- Q13. No internet assets: every page is served from the laptop (D12).
- SAM is tested on the laptop before any tool code (D3). Job control is a standalone module (D4).

Answered (Nick, chat 2026-11-25):
- Q15. Urchin type: SAM finds one class, "urchin" (it cannot tell purple from red; S1, D10).
  In label review, `P` or `R` tags each urchin purple or red. A frame cannot be accepted while
  it has an untagged urchin. Exports still have separate purple urchin and red urchin classes (D18).
- Q16. Viewing images large is the same everywhere in the app: clicking a frame opens it large on the
  same page (never a new tab), with < > buttons and arrow keys to move through the page's frames, Esc to
  close, and zoom and pan (S2's frame viewer). Pages that show masks add a class legend along the bottom
  of the viewer, where clicking a class shows or hides its masks.
- Q17. Classes are edited in a form (rows, choices and buttons, like Roboflow or Edge Impulse), not by
  typing YAML in a text box, which invites entry errors. `classes.yaml` stays the file on disk.
- Q18. A third urchin type, "black urchin" for now (the name may change later; it is clearly different from
  red and purple). It shares the "urchin" SAM prompt with purple and red and is tagged in label review like
  them. This replaces "No third urchin class now" under Q14. Most urchins are purple: review should find
  and tag the outliers, not make Nick click every urchin (D39). Open for the later packet milestone: the
  hourly packet in `docs/source/mvp_spec.md` has 2 urchin types (count 4 bytes, cover per type).

Answered (Nick, spec lead's chat 2026-11-26; Build 2, `docs/source/build2_spec.md`):
- Q19. Instance segmentation: trained and compared on the laptop; on a camera only if it converts,
  fits and runs in about a day, like the AE3.
- Q20. Licences: permissive only for every model in the app; start with YOLOX (Apache-2.0).
- Q21. Test frames may go to his own bench units over any link; never to a cloud.
- Q22. Bench units: his Pi Zero 2W rig (IMX708, N6, AE3) and his Pi 5 screen rig; Nick is the owner and runs bench tests from the app.
- Q23. Bench method: frames from file first; the screen-in-loop rig stays an option.
- Q24. Outside data and starting weights: the urchin datasets from his earlier detector work, already
  on the laptop with their licences recorded. Only sources whose recorded verdict allows commercial use.

Open for Build 2 (flag, don't guess; `docs/source/build2_spec.md` §Open questions): the Pi's runtime
and whether the bench Pi is 64-bit; a segmenter architecture that places fully on the N6's NPU; the
licence of each set of published starting weights; what else runs on the two bench units; whether a
validated power meter is wired on the Pi Zero 2W rig; whether weights trained on the DUO set may start a
camera model (for Nick); accepting ST's licence terms for ST Edge AI Core (for Nick).
