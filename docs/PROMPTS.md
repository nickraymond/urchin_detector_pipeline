# PROMPTS.md — Session Kickoff Prompts

*Copy-paste these verbatim; fill only `<N>` and `<name>`. Never add task
instructions here or in the prompt itself — requirements belong in SPEC.md or
TRACKER.md, where the next session can see them.*

*(No skills installed? Replace the first line of any prompt with: "Follow the
Rules for Agents at the top of docs/TRACKER.md.")*

---

## 1 — New sprint

```
Run /agent-entry. We're starting Sprint S<N>.

Read docs/TRACKER.md cover to cover, skim docs/SPEC.md and docs/DESIGN.md,
and read the top 3 entries of docs/DEV_LOG.md before doing anything else.

Then work through S<N>'s bites in order under "How this team works" in
CLAUDE.md: plan at the top of each draft PR, code + tests, self-test in the
browser, QE, the lead merges. Don't wait for me between bites.

This sprint is done when I've run its test build (the TRACKER Demo) and it passes.
```

## 2 — Resume mid-sprint

```
Run /agent-entry. We're mid-Sprint S<N> on branch sprint/<N>-<name>.
Do the full read ritual, then check the branch diff against the TRACKER
state and the top DEV_LOG entry, note where the last session
stopped and which nibble you're in, then carry on.
```

## 3 — Sprint close-out (demo already passed)

```
Sprint S<N> demo passed on my end. Close it out: mark S<N> done in
TRACKER.md, append the DEV_LOG entry, add any DESIGN.md decisions made
this sprint, and open the PR with the demo commands in the description.
```

## 4 — Capture a task without acting on it

```
Run /capture-task: <one-line description>. Size it, place it (current
sprint / later sprint / icebox), show me the TRACKER diff, and then
return to the current bite — do not start work on it.
```
