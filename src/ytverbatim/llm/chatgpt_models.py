"""Model families for the chatgpt provider: `gpt-sol` means the newest Sol model the ChatGPT plan lists.

A name shaped `gpt-<family>` is an alias, resolved once per run from the plan's model catalog. A name with a version,
`gpt-6.1-sol`, is a pin and is used as is. The server knows no alias, so nothing here falls back to a fixed version.
"""

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from ytverbatim.errors import ConfigError
from ytverbatim.llm import chatgpt_auth
from ytverbatim.llm.chatgpt_auth import LOGIN_COMMAND, ChatGPTAuthError
from ytverbatim.llm.chatgpt_http import Opener, error_code, https_open

MODELS_URL = "https://api.openai.com/v1/models"
_TIMEOUT_S = 30.0
_UNAUTHORIZED = 401
_ALIAS = re.compile(r"gpt-[a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)*")
_VERSIONED = re.compile(r"gpt-(?P<version>\d+(?:\.\d+)*)-(?P<family>[a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)*)")
_PIN_EXAMPLE = "gpt-6.2-sol"


@dataclass(frozen=True)
class _Entry:
    slug: str
    selectable: bool


def is_alias(model: str) -> bool:
    return _ALIAS.fullmatch(model) is not None


def resolve(model: str, *, opener: Opener = https_open, reauth: Callable[[], object] | None = None) -> str:
    """Return `model` itself when it is a pin, else the slug of the newest member of its family.

    Fails loudly, with a `fix` that says how to pin, when the catalog cannot be read or lists no member. With
    `reauth`, a catalog request refused as unauthorized signs in again once.
    """
    if not is_alias(model):
        return model
    return _newest(model, _catalog(model, opener, reauth))


def _newest(alias: str, entries: list[_Entry]) -> str:
    if any(entry.slug == alias for entry in entries):
        return alias
    family = alias.removeprefix("gpt-")
    members: list[tuple[tuple[int, ...], str]] = []
    for entry in entries:
        found = _VERSIONED.fullmatch(entry.slug)
        if entry.selectable and found is not None and found["family"] == family:
            members.append((tuple(int(part) for part in found["version"].split(".")), entry.slug))
    if not members:
        raise _error(f"the ChatGPT plan lists no {family!r} model, so {alias} has no newest member")
    return max(members)[1]


def _catalog(alias: str, opener: Opener, reauth: Callable[[], object] | None) -> list[_Entry]:
    try:
        token = chatgpt_auth.access_token(opener=opener)
        try:
            body = _get(token, opener)
        except urllib.error.HTTPError as exc:
            if exc.code != _UNAUTHORIZED or reauth is None:
                raise
            exc.close()
            reauth()
            body = _get(chatgpt_auth.access_token(opener=opener), opener)
    except ChatGPTAuthError as exc:
        raise _error(
            f"cannot read the ChatGPT model catalog to resolve {alias}: {exc}", signed_out=exc.needs_sign_in
        ) from None
    except urllib.error.HTTPError as exc:
        code = error_code(exc)
        status = f"HTTP {exc.code}" + (f", {code}" if code else "")
        if exc.code == _UNAUTHORIZED:
            raise _error(
                f"OpenAI rejected the ChatGPT sign-in ({status}) while resolving {alias}", signed_out=True
            ) from None
        raise _error(f"the ChatGPT model catalog could not be read to resolve {alias} ({status})") from None
    except TimeoutError:
        raise _error(f"the ChatGPT model catalog did not answer within {_TIMEOUT_S:g}s to resolve {alias}") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        raise _error(f"cannot reach OpenAI to resolve {alias} ({reason})") from None
    return _parse(alias, body)


def _get(token: str, opener: Opener) -> bytes:
    request = urllib.request.Request(MODELS_URL, headers={"Authorization": f"Bearer {token}"}, method="GET")
    response = opener(request, _TIMEOUT_S)
    try:
        return response.read()
    finally:
        response.close()


def _parse(alias: str, body: bytes) -> list[_Entry]:
    try:
        payload: object = json.loads(body)
    except ValueError:
        payload = None
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise _error(f"the ChatGPT model catalog has an unexpected shape, so {alias} cannot be resolved")
    entries: list[_Entry] = []
    for item in models:
        if isinstance(item, dict) and isinstance(slug := item.get("slug"), str):
            entries.append(_Entry(slug, item.get("visibility") == "list" and item.get("supported_in_api") is True))
    return entries


def _error(message: str, *, signed_out: bool = False) -> ConfigError:
    pin = f'pin a version: --llm chatgpt:{_PIN_EXAMPLE}, or [llm] model = "{_PIN_EXAMPLE}"'
    fix = f"run `{LOGIN_COMMAND}`, or {pin}" if signed_out else pin
    error = ConfigError(f"{message}; {fix}")
    error.fix = fix
    return error
