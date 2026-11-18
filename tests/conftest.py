import os
import urllib.request

import pytest

from anyverbatim import legacy
from anyverbatim.llm import chatgpt_http


@pytest.fixture(autouse=True)
def no_openai_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the real OpenAI opener raise, so no test can reach OpenAI even through a default argument."""

    def refuse(request: urllib.request.Request, *_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"a test tried to contact {request.full_url}")

    monkeypatch.setattr(chatgpt_http._OPENER, "open", refuse)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point HOME and the XDG roots at an empty directory and drop the app's variables.

    Every path lookup moves directories left under the old name, so a test reaching the real home could move the
    developer's own data.
    """
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    for name in list(os.environ):
        if name.startswith(("XDG_", legacy.ENV_PREFIX, legacy.LEGACY_ENV_PREFIX)):
            monkeypatch.delenv(name)
