# urchin_detector_pipeline

A local labeling tool for the Urchin Monitoring MVP: GoPro videos → frames → SAM labeling →
reviewed COCO boxes and per-pixel class masks. It runs in the browser on Nick's laptop;
images never leave the laptop.

Status: scaffold only (Sprint S0). No app code yet — see `docs/TRACKER.md`.

## Docs

- `docs/source/mvp_spec.md` — Nick's full MVP spec (the source)
- `docs/SPEC.md` — condensed contract: goal, facts, hard constraints, open questions
- `docs/TRACKER.md` — rules + sprint ladder (start here)
- `docs/DESIGN.md`, `docs/DEV_LOG.md`, `docs/PROMPTS.md` — decisions, session log, kickoff prompts

## Setup (so far)

```bash
cp config.example.toml config.local.toml
```

Then open `config.local.toml` and check the two paths. That file is gitignored. Videos,
frames, datasets and model weights never go in git.

Process: this repo was created from `template_LLM_project` (agent discipline in
`docs/TRACKER.md`; skills in `.claude/skills/`).
