# CLAUDE.md — urchin_detector_pipeline

## Start here, every session

This repo runs on the agent discipline in **docs/TRACKER.md**. Before any other
work: run **/agent-entry** (or follow the Rules for Agents at the top of
docs/TRACKER.md). Owner and approval gate: **Nick**.

Docs map — read per the ritual, don't skip it:

- `docs/SPEC.md` — goal; verified facts; hard constraints; open questions
- `docs/TRACKER.md` — rules + sprint ladder (the entry point)
- `docs/DESIGN.md` — as-built architecture + decision log
- `docs/DEV_LOG.md` — session log, newest first
- `docs/PROMPTS.md` — Nick's kickoff prompts

Layout: `app/` = the labeling tool (web UI, worker, one module per stage) · `tests/` ·
`docs/source/` = Nick's full MVP spec · data lives outside git (see `config.example.toml`).

## How this team works (Nick, 2026-11-25: approved all 7 changes)

Nick wrote the spec. The team works from it on its own and brings him working
things to test; he gives feedback on what he can click. This section wins over
any older step in this repo's docs, skills or prompts.

1. **Plans need no approval from Nick.** The builder writes the bite's plan at the
   top of its draft PR; QE checks it against SPEC, TRACKER and DESIGN; the builder
   carries on.
2. **Nick tests once per sprint.** At the end of each sprint the lead hands him one
   test build (the sprint's Demo in TRACKER) with a click-through checklist. A
   sprint may add at most one mid-sprint check, only when how a page looks needs
   his eye. His feedback becomes TRACKER items. Builders test each bite
   themselves in the browser and say in the PR what they saw.
3. **The planning lead merges** any PR with a QE PASS on its exact head
   (squash, `--match-head-commit`) and checks `main` afterwards. This standing
   grant replaces the per-session merge grant of the team rules and holds for
   successor lead sessions. Nick merges these himself: changes to `CLAUDE.md`,
   TRACKER's Rules for Agents, anything
   under `.claude/`, `docs/PROMPTS.md`; changes to SPEC's hard constraints; and
   changes to the spec's goal or scope.
4. **Gaps in the spec are Lead calls.** The lead decides within the spec's
   intent, logs each call in DESIGN's decision log as "Lead call", and lists
   them in the next test build's note. Nick can reverse any of them.
5. **Nick is asked only about:** a change to or contradiction of the spec or a
   hard constraint; anything that sends data off the laptop; a new model or
   library whose licence is not verified; money; hardware. Questions are saved
   up and sent together, each with a recommendation.
6. **No "confirm" or "wait for my go" steps.** Sessions take the next open
   TRACKER item. The lead starts and rotates builder and QE sessions.
7. **Sprint plans:** the lead writes them from the spec and QE reviews them.
   Nick sees a plan only if it would change the spec.
8. **Finished sessions are archived (Nick, 2026-11-25).** When a builder, QE or
   lead session rotates (handoff written, successor started) or finishes its
   last ticket, the planning lead archives it at once; the lead's own successor
   archives it. Before archiving, check that its worktree has no uncommitted
   work and that its PRs are merged or closed; if not, keep it and say why in
   the handoff. Never archive a session whose worktree a live session still
   uses. Archiving can be undone; deleting a session is never done.

Team rules (`~/Documents/GitHub/nereus-software-engineering-team/TEAM_RULES.md`)
still apply: QE before ready, approvals read at the source, never make it up.

## Engineering values (apply to every bite)

1. **Boring, debuggable engineering.** Small modules, explicit control flow,
   visible logs, plain formats. Build for the current sprint, not an imagined
   future.
2. **Reuse before rewriting.** Inspect prior art first; adapt the smallest
   working piece; document what was reused. A rewrite needs a measurable reason.
3. **Never invent facts.** APIs, formats, limits, pinouts: verify against
   primary sources or measure, else flag in SPEC.md §Open questions.
4. **Trust artifacts, not exit codes.** Verify outputs exist, sizes are
   plausible, and content is usable before calling anything done.
5. **One variable at a time.** Record the known-good path before changing it.
6. **Fail loudly and usefully.** Errors carry context and a recovery hint;
   partial failure never destroys good data.
7. **Data stays local and out of git.** Images never leave Nick's laptop; no tokens
   or secrets in git; videos, frames, datasets and model weights live in the data
   folder named in `config.local.toml`, never in the repo — docs/SPEC.md §Hard constraints.
8. **Hard constraints in SPEC.md are absolute.**

> Never trust a script just because it exits successfully. Trust the artifacts.
