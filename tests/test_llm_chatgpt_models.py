import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest
from chatgpt_fakes import FakeOpener, FakeResponse, Route, http_error, json_response

from anyverbatim.config import LLMOptions
from anyverbatim.errors import ConfigError
from anyverbatim.llm import ChatGPTClient, chatgpt, chatgpt_auth, client_from_options
from anyverbatim.llm.chatgpt_models import MODELS_URL, is_alias, resolve
from anyverbatim.llm.protocol import LLMRequest

SECRET = "base-instructions-that-must-never-be-printed"


def entry(slug: str, *, visibility: str = "list", api: bool = True) -> dict[str, object]:
    return {"slug": slug, "visibility": visibility, "supported_in_api": api, "base_instructions": SECRET}


def catalog(*entries: dict[str, object]) -> FakeOpener:
    return FakeOpener({f"GET {MODELS_URL}": json_response({"models": list(entries)})})


def listing(*slugs: str) -> FakeOpener:
    return catalog(*(entry(slug) for slug in slugs))


@pytest.fixture(autouse=True)
def signed_in(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    tokens: list[str] = []

    def access_token(*, opener: object) -> str:
        tokens.append("token")
        return f"token-{len(tokens)}"

    monkeypatch.setattr(chatgpt_auth, "access_token", access_token)
    return tokens


@pytest.mark.parametrize(
    ("model", "alias"),
    [
        ("gpt-sol", True),
        ("gpt-astra", True),
        ("gpt-reserve", True),
        ("gpt-sol-mini", True),
        ("gpt-6.1-sol", False),
        ("gpt-6-sol", False),
        ("gpt-5.6-sol", False),
        ("gpt-4o", False),
        ("sol", False),
        ("claude-sonnet-5-5", False),
    ],
)
def test_a_name_without_a_version_is_an_alias(model: str, alias: bool) -> None:
    assert is_alias(model) is alias


@pytest.mark.parametrize(
    ("slugs", "newest"),
    [
        (["gpt-6.1-sol", "gpt-6.2-sol", "gpt-6-sol", "gpt-5.6-sol"], "gpt-6.2-sol"),
        (["gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol"], "gpt-6.1-sol"),
        (["gpt-6.1-sol", "gpt-6-sol"], "gpt-6.1-sol"),
        (["gpt-6-sol", "gpt-5.6-sol"], "gpt-6-sol"),
        (["gpt-6.9-sol", "gpt-6.10-sol"], "gpt-6.10-sol"),
        (["gpt-6.2-sol", "gpt-6.2.1-sol"], "gpt-6.2.1-sol"),
        (["gpt-6.2-sol"], "gpt-6.2-sol"),
    ],
)
def test_the_newest_is_the_highest_numeric_version(slugs: list[str], newest: str) -> None:
    assert resolve("gpt-sol", opener=listing(*slugs)) == newest


def test_only_the_exact_family_counts() -> None:
    opener = listing("gpt-6.3-sol-mini", "gpt-6.2-astra", "gpt-6.1-sol", "gpt-6.4-solar", "gpt-7-sol2")
    assert resolve("gpt-sol", opener=opener) == "gpt-6.1-sol"
    assert resolve("gpt-sol-mini", opener=opener) == "gpt-6.3-sol-mini"
    assert resolve("gpt-astra", opener=opener) == "gpt-6.2-astra"


def test_hidden_and_non_api_models_are_skipped() -> None:
    opener = catalog(
        entry("gpt-6.4-sol", visibility="hide"),
        entry("gpt-6.3-sol", api=False),
        entry("gpt-6.2-sol"),
        {"slug": "gpt-6.5-sol", "visibility": "list"},
        entry("gpt-5.6-sol"),
    )
    assert resolve("gpt-sol", opener=opener) == "gpt-6.2-sol"


def test_a_slug_equal_to_the_alias_is_used_as_listed() -> None:
    opener = catalog(entry("gpt-reserve", visibility="hide"), entry("gpt-6.2-reserve"))
    assert resolve("gpt-reserve", opener=opener) == "gpt-reserve"


def test_a_pin_is_used_as_is_without_a_catalog_request(signed_in: list[str]) -> None:
    opener = FakeOpener()
    assert resolve("gpt-6.1-sol", opener=opener) == "gpt-6.1-sol"
    assert resolve("gpt-6-sol", opener=opener) == "gpt-6-sol"
    assert opener.requests == [] and signed_in == []


def test_the_catalog_request_carries_the_sign_in_and_nothing_else_is_asked() -> None:
    opener = listing("gpt-6.2-sol")
    resolve("gpt-sol", opener=opener)
    (request,) = opener.requests
    assert (request.get_method(), request.full_url, request.data) == ("GET", MODELS_URL, None)
    assert request.get_header("Authorization") == "Bearer token-1"


def failing(error: Exception) -> FakeOpener:
    def route(_request: urllib.request.Request) -> FakeResponse:
        raise error

    return FakeOpener({f"GET {MODELS_URL}": route})


def body(payload: bytes) -> FakeOpener:
    return FakeOpener({f"GET {MODELS_URL}": lambda _request: FakeResponse(body=payload)})


@pytest.mark.parametrize(
    ("opener", "said"),
    [
        (FakeOpener({f"GET {MODELS_URL}": http_error(500, {"error": "server_error"})}), "HTTP 500, server_error"),
        (FakeOpener({f"GET {MODELS_URL}": http_error(403, {"error": {"code": "missing_scope"}})}), "HTTP 403"),
        (failing(urllib.error.URLError("no route to host")), "cannot reach OpenAI to resolve gpt-sol"),
        (failing(ConnectionResetError("reset")), "cannot reach OpenAI"),
        (failing(TimeoutError()), "did not answer within 30s"),
        (body(b"not json " + SECRET.encode()), "unexpected shape"),
        (body(json.dumps({"data": [entry("gpt-6.2-sol")]}).encode()), "unexpected shape"),
        (body(json.dumps({"models": {"slug": "gpt-6.2-sol"}}).encode()), "unexpected shape"),
        (body(b"[]"), "unexpected shape"),
        (listing("gpt-6.2-astra", "gpt-6.2-sol-mini"), "lists no 'sol' model"),
        (catalog(entry("gpt-6.2-sol", visibility="hide"), entry("gpt-6.1-sol", api=False)), "lists no 'sol' model"),
        (catalog(), "lists no 'sol' model"),
        (body(json.dumps({"models": [None, 3, {"slug": 7}, {}]}).encode()), "lists no 'sol' model"),
    ],
)
def test_an_unreadable_catalog_or_an_empty_family_fails_loudly_with_a_fix(opener: FakeOpener, said: str) -> None:
    with pytest.raises(ConfigError, match=said) as raised:
        resolve("gpt-sol", opener=opener)
    error = raised.value
    assert '--llm chatgpt:gpt-6.2-sol, or [llm] model = "gpt-6.2-sol"' in error.fix
    assert error.fix in str(error)
    assert SECRET not in str(error) and "token-1" not in str(error)


def test_a_rejected_sign_in_says_to_log_in_or_pin() -> None:
    opener = FakeOpener({f"GET {MODELS_URL}": http_error(401, {"error": "invalid_token"})})
    with pytest.raises(ConfigError, match="rejected the ChatGPT sign-in") as raised:
        resolve("gpt-sol", opener=opener)
    assert raised.value.fix.startswith("run `anyverbatim login`, or pin a version")


def test_a_lapsed_sign_in_is_a_login_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def access_token(*, opener: object) -> str:
        raise chatgpt_auth.ChatGPTAuthError("not signed in to ChatGPT", code="not_signed_in")

    monkeypatch.setattr(chatgpt_auth, "access_token", access_token)
    with pytest.raises(ConfigError, match="not signed in to ChatGPT") as raised:
        resolve("gpt-sol", opener=FakeOpener())
    assert raised.value.fix.startswith("run `anyverbatim login`")


def test_an_unauthorized_catalog_request_signs_in_again_once_when_allowed(signed_in: list[str]) -> None:
    answers: list[Route] = [http_error(401), json_response({"models": [entry("gpt-6.2-sol")]})]
    opener = FakeOpener({f"GET {MODELS_URL}": lambda request: answers.pop(0)(request)})
    reauths: list[str] = []
    assert resolve("gpt-sol", opener=opener, reauth=lambda: reauths.append("again")) == "gpt-6.2-sol"
    assert reauths == ["again"]
    assert [r.get_header("Authorization") for r in opener.requests] == ["Bearer token-1", "Bearer token-2"]


def test_a_second_unauthorized_answer_is_an_error_after_one_sign_in() -> None:
    opener = FakeOpener({f"GET {MODELS_URL}": http_error(401)})
    reauths: list[str] = []
    with pytest.raises(ConfigError, match="rejected the ChatGPT sign-in"):
        resolve("gpt-sol", opener=opener, reauth=lambda: reauths.append("again"))
    assert reauths == ["again"] and len(opener.requests) == 2


def test_a_failed_sign_in_again_is_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def reauth() -> None:
        raise chatgpt_auth.ChatGPTAuthError("sign-in timed out", code="timeout")

    opener = FakeOpener({f"GET {MODELS_URL}": http_error(401)})
    with pytest.raises(ConfigError, match="sign-in timed out"):
        resolve("gpt-sol", opener=opener, reauth=reauth)


def test_the_error_body_of_a_refused_request_is_never_echoed() -> None:
    refused = urllib.error.HTTPError(
        MODELS_URL, 400, "bad", Message(), io.BytesIO(json.dumps({"error": f"bad {SECRET}"}).encode())
    )
    with pytest.raises(ConfigError, match="HTTP 400") as raised:
        resolve("gpt-sol", opener=failing(refused))
    assert SECRET not in str(raised.value)


def test_the_default_client_is_the_newest_sol_and_its_identity_carries_the_slug() -> None:
    opener = listing("gpt-6.1-sol", "gpt-6.2-sol", "gpt-6.2-astra")
    default = client_from_options(LLMOptions(), opener=opener)
    assert default.id == "chatgpt:gpt-6.2-sol@medium"
    assert client_from_options(LLMOptions(effort="high"), opener=opener).id == "chatgpt:gpt-6.2-sol@high"
    astra = client_from_options(LLMOptions(model="gpt-astra", effort="low"), opener=opener)
    assert astra.id == "chatgpt:gpt-6.2-astra@low"


def test_a_pinned_model_builds_a_client_without_a_catalog_request() -> None:
    opener = FakeOpener()
    pinned = client_from_options(LLMOptions(model="gpt-6.1-sol"), opener=opener)
    assert pinned.id == "chatgpt:gpt-6.1-sol@medium"
    assert opener.requests == []


def test_every_call_of_a_client_uses_the_one_resolved_slug() -> None:
    opener = listing("gpt-6.2-sol")
    resolved = client_from_options(LLMOptions(), opener=opener)
    assert isinstance(resolved, ChatGPTClient)
    request = LLMRequest(system="s", user="u")
    assert [resolved.payload(request)["model"] for _ in range(3)] == ["gpt-6.2-sol"] * 3
    assert len(opener.to(MODELS_URL)) == 1


def test_a_newer_model_changes_the_identity_that_checkpoints_are_keyed_by() -> None:
    before = client_from_options(LLMOptions(), opener=listing("gpt-6.1-sol"))
    after = client_from_options(LLMOptions(), opener=listing("gpt-6.1-sol", "gpt-6.2-sol"))
    assert before.id != after.id


def test_the_catalog_failure_surfaces_when_the_client_is_built() -> None:
    with pytest.raises(ConfigError, match="lists no 'sol' model"):
        client_from_options(LLMOptions(), opener=listing("gpt-6.2-astra"))


def test_the_default_model_is_an_alias() -> None:
    assert (chatgpt.DEFAULT_MODEL, is_alias(chatgpt.DEFAULT_MODEL)) == ("gpt-sol", True)
