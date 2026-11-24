# SPEC.md — urchin_detector_pipeline

*What Nick wants. Stable reference — agents skim this; changes require Nick's approval.*
*Last updated: 2026-11-24*

## Goal

A local labeling tool, run in the browser on Nick's laptop, that turns a few GoPro videos
into reviewed training data for two on-camera models: COCO boxes for a detector and
per-pixel class masks for a segmenter. Done when Nick can start the app with one command,
walk six stages in the UI (ingest → pick clips → extract frames → review frames → SAM
labeling → review + export) on his ReefCheck videos (or on folders of still images), and the export passes the acceptance
criteria in `docs/source/mvp_spec.md`.

## Background

- Full spec: `docs/source/mvp_spec.md` ("Urchin Monitoring MVP — Spec", Nick, 2026-11-24).
  That file is the source; this file is the condensed contract. If they disagree, stop and
  ask Nick.
- MVP purpose: choose the go-to-market camera (OpenMV N6 vs Raspberry Pi) from a one-day
  dawn-to-dusk bake-off measuring urchins, sea stars and kelp. This repo is Build 1 only
  (the labeling tool).
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

1. **Images never leave the laptop.** No cloud APIs and no uploads. Only exception: Claude
   coding sessions may view frames (Nick, 2026-11-24).
2. **No tokens or secrets in git.** Hugging Face access only through `hf auth login`; the
   app never stores the token.
3. **Videos, frames, datasets and model weights stay out of git**, in the data folder named
   in `config.local.toml` (on the SSD). Source code lives in git.
4. **Source videos are read-only.** Never modify, move or delete them.
5. **Delete means move to `rejected/`**, never erase.
6. **Licenses:** on-camera models permissive only (Apache / MIT). The SAM License is
   acceptable for the labeling tool only.
7. **Every stage has a browser UI.** The terminal is used only to start the app. Basic and
   functional, not flashy (Nick, chat 2026-11-24).
8. **The repo is public** (Nick, chat 2026-11-24). No real frames, video stills, people's
   names or tokens in git, PRs, issues or test fixtures. Tests use synthetic images.

## Success criteria by sprint

See TRACKER.md — every sprint ends with a live demo Nick can run.

## Non-goals (this project)

- Training, quantizing, camera firmware, the hourly Iridium packet (later milestones).
- Hand-editing mask outlines; colour correction of frames.
- Panoptic export; tracking individual animals; kelp growth rates in cm/day.
- Multi-user, cloud, login.
- Kelp and kelp shoot classes, until Nick has reference images (then a settings change).

## Open questions (flag, don't guess)

- Q2. Overlap rules for area-vs-area and for several prompts of one class.
- Q6. Which sprint carries "Suggest clips" and "Pre-screen with SAM".
- Q9. CVAT export format for fine outline fixes, and whether fixes are imported back.
- Q10. Defaults: blur threshold, hash distance, minimum mask area, inference resolution,
  tile size, memory guard.
- Q11. "Largest object %" by mask or by box; the id for unknown; which frames count as
  "accepted" for export.
- Q12. Does SAM 3.1 run on the M1 Max? A read of the `facebookresearch/sam3` readme on
  2026-11-24 reported these prerequisites: Python 3.12+, PyTorch 2.7+, and a CUDA GPU; it
  does not mention Apple MPS or CPU. The Hugging Face page says the 3.1 repo has checkpoints
  only, with no Transformers integration. Both to be re-read and tested in S1. Also open:
  the SAM License terms (the repo's `LICENSE` file has not been read yet).
- Q13. May the UI load assets from the internet (a CDN)? The planning lead's reading of
  constraint 1 is no; Nick to confirm.

Answered (Nick, chat 2026-11-24):
- Repo name stays `urchin_detector_pipeline`; the repo is public.
- Project data lives on the SSD; source code in git (DESIGN D1).
- Laptop is an Apple M1 Max, 32 GB. Agents may view real frames.
- Q1. Kelp and kelp shoot wait for reference images; adding them later is a settings change (D5).
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
- SAM is tested on the laptop before any tool code (D3). Job control is a standalone module (D4).
