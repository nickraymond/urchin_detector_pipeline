"""Build 2, "what runs where, on the boards" (S7; DESIGN §Planned for S7, D81).

    units.py    the bench units listed under [bench] in config.local.toml; anything else is refused
    board.py    the `Board` interface (info, load, time, load-several, reset) and `FakeBoard`
    riglog.py   <data_dir>/bench/rig_log.jsonl: every change on the rig, before and after, with its undo
    pages.py    the "Rig log" page

Builders and QE use the fake board only (SPEC constraint 9); real boards come with bites 3-5.
"""
