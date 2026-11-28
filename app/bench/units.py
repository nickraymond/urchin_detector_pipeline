"""Bench units (D81, SPEC constraint 9). They are listed only in config.local.toml, never in git:

    [bench]
    field_units = ["<a field unit's name>"]     # refused by name, always
    [bench.units.<unit name>]
    ssh_host = "<alias in ~/.ssh/config>"
    cameras = ["n6", "ae3"]                     # which camera kinds hang on that Pi's USB

With no [bench] table only the fake board exists. A unit that is not listed, or whose name or ssh host
is a field unit (compared case-blind, without user@ or :port), is refused with a reason. A [bench] table with
no field_units or a mistyped key uses no unit at all. Nothing here ever opens a connection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FAKE = "fake"                      # the one unit that always exists: FakeBoard, for builders and QE
CAMERA_KINDS = ("n6", "ae3")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Refused(Exception):
    """A unit the app must not touch. The message says why and what to change."""


@dataclass(frozen=True)
class Unit:
    name: str
    ssh_host: str = ""
    cameras: tuple[str, ...] = ()

    @property
    def fake(self) -> bool:
        return self.name == FAKE

    @property
    def kinds(self) -> tuple[str, ...]:
        """The boards on this unit: the Pi itself, then its cameras."""
        return ("pi", *self.cameras)


@dataclass
class Bench:
    units: dict[str, Unit] = field(default_factory=dict)
    field_units: frozenset[str] = frozenset()   # the refusal set: each entry in its full and short form
    field_count: int = 0                         # entries as written (QE PR 98 nit 1)
    problems: list[str] = field(default_factory=list)   # shown on the page; a bad entry is left out
    n6_dfu_serial: str = ""                     # the bench N6's bootloader serial (SPEC: never in git); "" = no N6 writes

    def unit(self, name: str) -> Unit:
        """The unit to run on, or Refused. The fake unit always exists and touches nothing."""
        if name == FAKE:
            return Unit(FAKE, cameras=CAMERA_KINDS)
        if _is_field(name, self.field_units):
            raise Refused(f"{name} is a field unit: the app never touches field units (SPEC constraint 9).")
        if name not in self.units:
            listed = ", ".join(sorted(self.units)) or "none"
            raise Refused(f"{name} is not a bench unit listed under [bench] in config.local.toml "
                          f"(listed: {listed}). Only the fake board can be used until a unit is listed.")
        return self.units[name]

    def names(self) -> list[str]:
        return [FAKE, *sorted(self.units)]


BENCH_KEYS = {"field_units", "units", "n6_dfu_serial"}
SERIAL = re.compile(r"^[0-9A-Fa-f]{8,64}$")
UNIT_KEYS = {"ssh_host", "cameras"}


def _key(name: str) -> str:
    """How names are compared with field_units: case-blind, no user@, no :port, and the first label of a
    dotted host name also counts ("pi@Field-Pi.local:22" is checked as "field-pi.local" and "field-pi")."""
    host = name.strip().rsplit("@", 1)[-1].split(":", 1)[0].strip().lower()
    return host


def _is_field(name: str, field_units: frozenset[str]) -> bool:
    k = _key(name)
    return k in field_units or k.split(".", 1)[0] in field_units


def parse(raw) -> Bench:
    """The [bench] table as read from config.local.toml (None when absent). When it is present, anything
    that could weaken the field-unit refusal (no field_units, a mistyped key) leaves every unit out."""
    bench = Bench()
    if raw is None:
        return bench
    if not isinstance(raw, dict):
        bench.problems.append("[bench] in config.local.toml must be a table; no bench unit is used.")
        return bench
    unknown = sorted(set(raw) - BENCH_KEYS)
    if unknown:
        bench.problems.append(f"[bench] has unknown key(s) {', '.join(unknown)} (known: field_units, units, "
                              "n6_dfu_serial); "
                              "no bench unit is used until it is fixed.")
        return bench
    if "field_units" not in raw:
        bench.problems.append('[bench] has no field_units list; no bench unit is used until it has one '
                              '(write field_units = [] only if there is no field unit at all).')
        return bench
    fields = raw["field_units"]
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
        # Without a trustworthy refusal list no real unit is safe to use.
        bench.problems.append("[bench] field_units must be a list of names; no bench unit is used until it is fixed.")
        return bench
    # Each entry counts by its full form and by its first label, so "field-pi.local" also refuses "field-pi".
    bench.field_units = frozenset(k for f in fields if f.strip() for k in (_key(f), _key(f).split(".", 1)[0]))
    bench.field_count = len({_key(f) for f in fields if f.strip()})
    serial = raw.get("n6_dfu_serial", "")
    if isinstance(serial, str) and SERIAL.match(serial.strip()):
        bench.n6_dfu_serial = serial.strip()
    elif serial != "":
        bench.problems.append("[bench] n6_dfu_serial must be the N6's DFU serial (hex digits, from the rig backup's "
                              "dfu_list.txt); the N6 route writes nothing until it is fixed.")
    units = raw.get("units", {})
    if not isinstance(units, dict):
        bench.problems.append("[bench.units] must be a table of units; none is used.")
        return bench
    for name, body in units.items():
        why = _problem(name, body, bench.field_units)
        if why:
            bench.problems.append(f"Bench unit {name!r} left out: {why}")
            continue
        bench.units[name] = Unit(name, body["ssh_host"].strip(), tuple(body.get("cameras", [])))
    return bench


def _problem(name, body, field_units) -> str:
    if name.lower() == FAKE:
        return f'"{FAKE}" is the fake board\'s name; pick another.'
    if not NAME.match(name):
        return "a name is letters, digits, '.', '_' or '-'."
    if not isinstance(body, dict):
        return "it must be a table with ssh_host and cameras."
    unknown = sorted(set(body) - UNIT_KEYS)
    if unknown:
        return f"unknown key(s) {', '.join(unknown)} (known: ssh_host, cameras)."
    host = body.get("ssh_host")
    if not isinstance(host, str) or not host.strip():
        return "it has no ssh_host (an alias from ~/.ssh/config)."
    if _is_field(name, field_units) or _is_field(host, field_units):
        return "it is listed in field_units; the app never touches field units."
    cams = body.get("cameras", [])
    if not isinstance(cams, list) or any(c not in CAMERA_KINDS for c in cams) or len(set(cams)) != len(cams):
        return f"cameras must be a list of {', '.join(CAMERA_KINDS)}, each at most once."
    return ""
