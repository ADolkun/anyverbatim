"""What carries over from the old name, ytverbatim: its directories, environment variables and fingerprint keys.

A directory left under the old name is moved into place on first use, never copied and never merged. An old variable
that is set without its new counterpart is an error rather than being read, so a setting is never silently ignored.
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from anyverbatim.errors import ConfigError

APP = "anyverbatim"
LEGACY_APP = "ytverbatim"
ENV_PREFIX = APP.upper() + "_"
LEGACY_ENV_PREFIX = LEGACY_APP.upper() + "_"


def check_environment(environ: Mapping[str, str] | None = None) -> None:
    """Raise `ConfigError` for each `YTVERBATIM_*` variable set without its `ANYVERBATIM_*` counterpart."""
    env = os.environ if environ is None else environ
    stale = sorted(
        name
        for name in env
        if name.startswith(LEGACY_ENV_PREFIX) and ENV_PREFIX + name.removeprefix(LEGACY_ENV_PREFIX) not in env
    )
    if not stale:
        return
    renamed = [ENV_PREFIX + name.removeprefix(LEGACY_ENV_PREFIX) for name in stale]
    error = ConfigError(
        f"{', '.join(stale)} {'is' if len(stale) == 1 else 'are'} set, but {LEGACY_APP} was renamed to {APP} and "
        f"reads {', '.join(renamed)} instead"
    )
    error.fix = f"rename the variable to {', '.join(renamed)} wherever you set it (shell profile, launchd, cron)"
    raise error


def move_into_place(old: Path, new: Path) -> Path:
    """Return `new`, after moving `old` there if only `old` exists; raise `ConfigError` if both do.

    The move is one `rename`, so a concurrent run sees either the old or the new directory, never half of one. The
    run that loses the race finds `old` gone and `new` present, and carries on.
    """
    # `new` is looked at first: a winner's rename makes `old` vanish and `new` appear at once, so seeing `new` and
    # then still seeing `old` means both really exist.
    if os.path.lexists(new):
        if os.path.lexists(old):
            raise _both_exist(old, new)
        return new
    if not os.path.lexists(old):
        return new
    try:
        old.rename(new)
    except FileNotFoundError:
        if os.path.lexists(new) and not os.path.lexists(old):
            return new
        raise _move_failed(old, new, "it vanished during the move") from None
    except OSError as exc:
        if os.path.lexists(new) and os.path.lexists(old):
            raise _both_exist(old, new) from exc
        raise _move_failed(old, new, exc.strerror or str(exc)) from exc
    # A host app embedding the library may run with no stderr at all (a windowed app).
    if sys.stderr is not None:
        sys.stderr.write(f"{APP}: moved {old} to {new} ({LEGACY_APP} was renamed to {APP})\n")
    return new


def fingerprint_factory(factory: str) -> str:
    """Spell an adapter's dotted path as fingerprints have always held it, under the old package name.

    Worker folders and stage keys digest the request fingerprint, which holds this path; keeping the old spelling
    keeps every transcript, diarization and LLM result made before the rename current.
    """
    package, dot, rest = factory.partition(".")
    return LEGACY_APP + dot + rest if package == APP and dot else factory


def _both_exist(old: Path, new: Path) -> ConfigError:
    error = ConfigError(f"both {old} and {new} exist; {LEGACY_APP} was renamed to {APP}, which uses only {new}")
    error.fix = f"merge {old} into {new}, or remove one of them, then run again"
    return error


def _move_failed(old: Path, new: Path, reason: str) -> ConfigError:
    error = ConfigError(f"cannot move {old} to {new} ({LEGACY_APP} was renamed to {APP}): {reason}")
    error.fix = f"move {old} to {new} yourself (for example with `mv`), then run again"
    return error
