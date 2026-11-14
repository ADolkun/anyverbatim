import urllib.request

import pytest

from ytverbatim.llm import chatgpt_http


@pytest.fixture(autouse=True)
def no_openai_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the real OpenAI opener raise, so no test can reach OpenAI even through a default argument."""

    def refuse(request: urllib.request.Request, *_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"a test tried to contact {request.full_url}")

    monkeypatch.setattr(chatgpt_http._OPENER, "open", refuse)
