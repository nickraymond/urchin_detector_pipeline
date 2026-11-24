# SPEC.md — urchin_detector_pipeline

*What Nick wants. Stable reference — agents skim this; changes require Nick's approval.*
*Last updated: 2026-11-24*

## Goal

A local labeling tool, run in the browser on Nick's laptop, that turns a few GoPro videos
into reviewed training data for two on-camera models: COCO boxes for a detector and
per-pixel class masks for a segmenter. Done when Nick can start the app with one command,
walk six stages in the UI (ingest → pick clips → extract frames → review frames → SAM
labeling → review + export) on his ReefCheck videos, and the export passes the acceptance
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
- The laptop has ffmpeg / ffprobe and Python 3.13.5; `uv` and the `hf` tool are not
  installed (checked 2026-11-24).
- Nick's Hugging Face account is approved for `facebook/sam3.1` (Nick, chat 2026-11-24).
- Coding agents may view real frames (Nick, chat 2026-11-24).

## Safety / hard constraints (non-negotiable)

1. **Images never leave the laptop.** No cloud APIs, no uploads, no remote UI assets (no
   CDN). Only exception: Claude coding sessions may view frames (Nick, 2026-11-24).
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

## Success criteria by sprint

See TRACKER.md — every sprint ends with a live demo Nick can run.

## Non-goals (this project)

- Training, quantizing, camera firmware, the hourly Iridium packet (later milestones).
- Hand-editing mask outlines; colour correction of frames.
- Panoptic export; tracking individual animals.
- Multi-user, cloud, login.

## Open questions (flag, don't guess)

- Q1. Kelp shoot is missing from the full spec's top class list; which segmenter class do
  its pixels get?
- Q2. Overlap rules for area-vs-area (kelp vs rock) and for several prompts of one class.
- Q3. Is the 300-frame cap applied before or after the blur / duplicate filters; what does
  "count matches the preview" compare?
- Q4. Frame names when two videos share a name; the chapter rule for non-GoPro names.
- Q5. 70 / 20 / 10 split by video when there are only a few videos.
- Q6. Which sprint carries "Suggest clips" and "Pre-screen with SAM" (milestone 1 says no SAM).
- Q7. What happens to results and review edits when a prompt or threshold changes.
- Q8. Scheduled start when the laptop is asleep or the lid is closed.
- Q9. How to add an object SAM missed; CVAT export format and re-import.
- Q10. Defaults: blur threshold, hash distance, minimum mask area, inference resolution,
  tile size, memory guard.
- Q11. "Largest object %" by mask or by box; the id for unknown; which frames count as
  "accepted" for export.
- Q12. SAM facts to verify against primary docs before S2: `facebook/sam3.1` has no
  Transformers support; Python / PyTorch needs of `facebookresearch/sam3`; Apple MPS
  support; SAM License terms; exact `hf` login commands.

Answered (Nick, chat 2026-11-24):
- Repo name stays `urchin_detector_pipeline`.
- Project data lives on the SSD; source code in git (DESIGN D1).
- Laptop is an Apple M1 Max, 32 GB.
- Agents may view real frames.
