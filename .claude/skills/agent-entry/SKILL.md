---
name: agent-entry
description: Session-start ritual for this repo's agent discipline. Use at the START of every coding session in a repo that has docs/TRACKER.md, or when asked to begin work, pick up a sprint, or continue the project. Enforces read order, bite sizing, nibble gates, branching, and demo-per-sprint.
---

# Agent Entry

You are working under this repo's agent discipline. OWNER (the human gate) is:
**Nick**.

## On session start — in this order, before any code

1. `git fetch origin` and check your branch against `origin/main` (in a worktree,
   `main` is checked out elsewhere, so do not `git checkout main`). Install any new
   skills the repo ships in `.claude/skills/`.
2. Read `docs/TRACKER.md` **cover to cover**. Non-negotiable.
3. Skim `docs/SPEC.md` and `docs/DESIGN.md`.
4. Read the top ~3 entries of `docs/DEV_LOG.md`.
5. Identify the single active TODO (first `[~]`, else first `[ ]` whose
   dependencies are met). Start it; do not wait for confirmation.
6. Read "How this team works" in `CLAUDE.md`: it wins over any older step here.

## How work happens

- **Bites:** ~300 LoC target, one TODO at a time. If SPEC.md is too thin to
  inform the bite, ask the planning lead for a Lead call — do not invent
  requirements. OWNER is asked only about the items in `CLAUDE.md`.
- **Nibbles (four, no OWNER gate):**
  1. **Plan** — at the top of the draft PR; QE checks it against the docs.
  2. **Code + unit tests.**
  3. **Self-test** — run the bite in the browser; write in the PR what you saw.
  4. **PR to QE** — the planning lead merges on QE PASS.
- **Branch:** all new work on `sprint/<n>-<name>`. Never commit to main.
- **Demo:** a sprint is not done until OWNER runs its test build (the sprint's
  Demo, handed over by the lead) successfully.

## On session end — always

- Add a DEV_LOG.md entry (newest on top): done / broke / next.
- Update DESIGN.md if architecture changed or a decision was made (append to
  the decision log; never rewrite history).
- Update TRACKER.md state markers.

## Standing rules

- Facts carry sources. Unknowns get flagged in SPEC.md §Open questions — never
  guessed. Model names, licenses, library APIs and file formats are verified
  against primary docs before use.
- Obey SPEC.md §Safety / hard constraints absolutely.
