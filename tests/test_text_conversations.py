"""Conversation-aware and per-stretch-language prompts, and the pinned prompts of undiarized YouTube documents."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from anyverbatim.acquire.captions import CaptionKind, CaptionSegment, CaptionTrack
from anyverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from anyverbatim.text.correct import CorrectionContext, correct_segments
from anyverbatim.text.numeric import Evidence, TimedText, verify_numbers
from anyverbatim.text.punctuate import punctuate

CLIENT_ID = "fake:conversation"
ZH_BARE = "大家好欢迎回来今天我们聊聊个股的估值问题"
EN_BARE = "so today we are going to talk about how the market reacts when the fed changes its mind"


@dataclass
class EchoClient:
    """Punctuates by appending a full stop, fixes nothing and picks the last reading; records every request."""

    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])
    respond: Callable[[LLMRequest], object] | None = None

    @property
    def id(self) -> str:
        return CLIENT_ID

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if '"items"' in request.user:
            items = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            reply: object = {"items": [{"i": it["i"], "text": it["text"] + "."} for it in items]}
        elif "SEGMENTS:" in request.user:
            reply = {"fixed": list[object]()}
        else:
            items = json.loads(request.user.split("ITEMS:\n", 1)[1])
            reply = {
                "verdicts": [
                    {"id": it["id"], "value": it["readings"][-1], "confidence": "high", "reason": "context"}
                    for it in items
                ]
            }
        return LLMResponse(text=json.dumps(reply, ensure_ascii=False), usage=Usage(10, 2), elapsed_s=0.0)


def items(request: LLMRequest) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
    return loaded


def digest(request: LLMRequest) -> str:
    return hashlib.sha256(json.dumps([request.system, request.user], ensure_ascii=False).encode()).hexdigest()


def checkpoints(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


YOUTUBE_TEXTS = [
    ZH_BARE,
    "today we cover the q4 results and what they mean for the rest of the year",
    "收入三万五千美元",
]
YOUTUBE_SEGMENTS = [TimedText("这只股票上周涨了十五个百分点，非常强劲", 0.0, 4.0)]
YOUTUBE_CAPTIONS = CaptionTrack(
    language="zh",
    kind=CaptionKind.MANUAL,
    source_format="json3",
    segments=(CaptionSegment(start=0.0, end=4.0, text="这只股票上周涨了五十个百分点非常强劲"),),
)


def youtube_requests(tmp_path: Path) -> tuple[EchoClient, dict[str, list[str]]]:
    client = EchoClient()
    punctuate(YOUTUBE_TEXTS, language="zh", client=client, checkpoint_dir=tmp_path / "punctuate")
    correct_segments(
        YOUTUBE_TEXTS,
        client=client,
        language="zh",
        context=CorrectionContext(glossary=("ETF", "GAAP"), creator_note="A pinned note."),
        checkpoint_dir=tmp_path / "correct",
        confidence=[[0.3, 0.4], None, [0.9]],
    )
    evidence = Evidence(primary="asr:w-prod", words=(), captions=(YOUTUBE_CAPTIONS,))
    verify_numbers(YOUTUBE_SEGMENTS, language="zh", evidence=evidence, client=client, checkpoint_dir=tmp_path / "num")
    keys = {stage: checkpoints(tmp_path / stage) for stage in ("punctuate", "correct", "num")}
    return client, keys


def test_undiarized_youtube_prompts_and_checkpoint_keys_are_pinned(tmp_path: Path) -> None:
    client, keys = youtube_requests(tmp_path)
    # Computed on main before speakers, stretch languages and the glossary reached the text stages.
    assert [digest(request) for request in client.requests] == [
        "52f67f48eda54b5b214cfa0de49381fc154b8ca7a111d87ff7f18e3cb1759bf1",
        "76e05ed9a6d42d049ef7c2ced97e9bb45219c5906a8be5e6b065b90349012ec4",
        "225951807d9c0aeab3856f29c16791d1111644a2fa0d81eac5c9519eb87be34c",
        "7d6d21d98fe8d66f86aeec5360bec0ad7dd28bffffc9438ea8d0d878c9fa3e83",
        "7b2adce3103d7ac98937b7c2c1e48b93755bb8bb33f48c89eaa4c47a7fc7616b",
    ]
    assert keys == {
        "punctuate": [
            "0138bf234bea4cade4e0940ca2a85bfb8f50aee3a711355db3a9b59d149afc7d.json",
            "26312d8cbe39a1d451f84777d96bd72297d915a7e93f1e0197864e1b16d9511b.json",
        ],
        "correct": [
            "d52be4886de58cba2b915d45b75604551512bbb1a82cc2b9095d28075e4bcccd.json",
            "fc04a655e6236ebd6663e4b6075b6c98bc6f0a16878b9fede6e391b324c5ff04.json",
        ],
        "num": ["dd840a831854ea302c64e075afa1100d189039d98aa6bcbd776dd6b6493b4e4b.json"],
    }


def test_passing_undiarized_single_language_stretch_data_keeps_the_youtube_prompts(tmp_path: Path) -> None:
    plain, plain_keys = youtube_requests(tmp_path / "plain")
    client = EchoClient()
    nones = [None] * len(YOUTUBE_TEXTS)
    punctuate(
        YOUTUBE_TEXTS,
        language="zh",
        client=client,
        checkpoint_dir=tmp_path / "given" / "punctuate",
        languages=nones,
        speakers=nones,
    )
    correct_segments(
        YOUTUBE_TEXTS,
        client=client,
        language="zh",
        context=CorrectionContext(glossary=("ETF", "GAAP"), creator_note="A pinned note."),
        checkpoint_dir=tmp_path / "given" / "correct",
        confidence=[[0.3, 0.4], None, [0.9]],
        languages=["zh", None, "zh"],
        speakers=nones,
    )
    evidence = Evidence(primary="asr:w-prod", words=(), captions=(YOUTUBE_CAPTIONS,))
    verify_numbers(
        YOUTUBE_SEGMENTS,
        language="zh",
        evidence=evidence,
        client=client,
        checkpoint_dir=tmp_path / "given" / "num",
        languages=[None],
    )
    assert [digest(r) for r in client.requests] == [digest(r) for r in plain.requests]
    given_keys = {stage: checkpoints(tmp_path / "given" / stage) for stage in ("punctuate", "correct", "num")}
    assert given_keys == plain_keys


def test_a_diarized_batch_carries_speaker_ids_and_says_it_is_a_conversation(tmp_path: Path) -> None:
    client = EchoClient()
    texts = [EN_BARE, "yeah and what did they do after that when the numbers came in late"]
    punctuate(
        texts, language="en", client=client, checkpoint_dir=tmp_path / "p", speakers=["S1", "S2"], languages=[None] * 2
    )
    correct_segments(
        texts,
        client=client,
        language="en",
        context=CorrectionContext(),
        checkpoint_dir=tmp_path / "c",
        speakers=["S1", "S2"],
    )
    punctuation, correction = client.requests
    assert items(punctuation)[1] == {"i": 1, "speaker": "S2", "before": "fed changes its mind", "text": texts[1]}
    assert items(correction) == [
        {"i": 0, "speaker": "S1", "text": texts[0]},
        {"i": 1, "speaker": "S2", "text": texts[1]},
    ]
    for request in (punctuation, correction):
        assert "turns of a conversation" in request.system


def test_an_undiarized_batch_has_no_speaker_field_and_no_conversation_sentence(tmp_path: Path) -> None:
    client = EchoClient()
    punctuate([EN_BARE], language="en", client=client, checkpoint_dir=tmp_path / "p", speakers=[None])
    correct_segments(
        [EN_BARE], client=client, language="en", context=CorrectionContext(), checkpoint_dir=tmp_path / "c"
    )
    for request in client.requests:
        assert "speaker" not in request.user
        assert "conversation" not in request.system


def test_each_segment_is_punctuated_by_its_own_languages_rules(tmp_path: Path) -> None:
    client = EchoClient()
    # An English document with one Mandarin stretch, and a Mandarin document with one English stretch.
    punctuate([ZH_BARE, EN_BARE], language="en", client=client, checkpoint_dir=tmp_path / "en", languages=["zh", None])
    punctuate([ZH_BARE, EN_BARE], language="zh", client=client, checkpoint_dir=tmp_path / "zh", languages=[None, "en"])
    for request in client.requests:
        (item,) = items(request)
        rules = "full-width Chinese punctuation" if item["text"] == ZH_BARE else "Use ASCII punctuation"
        assert rules in request.system
    # A neighbour's edge is cut by the neighbour's own language: sixteen Mandarin characters, four English words.
    english_in_en = items(client.requests[1])[0]
    assert english_in_en["text"] == EN_BARE
    assert english_in_en["before"] == ZH_BARE[-16:]
    mandarin_in_zh = items(client.requests[2])[0]
    assert mandarin_in_zh["text"] == ZH_BARE
    assert mandarin_in_zh["after"] == "so today we are"


def test_a_mixed_language_transcript_says_so_to_correction_and_to_the_numeric_verdicts(tmp_path: Path) -> None:
    client = EchoClient()
    correct_segments(
        [ZH_BARE, EN_BARE],
        client=client,
        language="zh",
        context=CorrectionContext(),
        checkpoint_dir=tmp_path / "c",
        languages=[None, "en"],
    )
    evidence = Evidence(primary="asr:w-prod", words=(), captions=(YOUTUBE_CAPTIONS,))
    verify_numbers(
        YOUTUBE_SEGMENTS,
        language="zh",
        evidence=evidence,
        client=client,
        checkpoint_dir=tmp_path / "n",
        languages=["en"],
    )
    line = "The transcript is mainly zh, with en mixed in; keep every word in the language it was spoken. "
    correction, verdicts = client.requests
    assert line in correction.user
    assert "The transcript language is" not in correction.user
    assert verdicts.user.startswith(line)


def test_per_text_languages_and_speakers_must_cover_every_text(tmp_path: Path) -> None:
    client = EchoClient()
    with pytest.raises(ValueError, match="speakers covers 1 texts, but there are 2"):
        punctuate([EN_BARE, EN_BARE], language="en", client=client, checkpoint_dir=tmp_path, speakers=["S1"])
    with pytest.raises(ValueError, match="languages covers 1 segments, but there are 2"):
        correct_segments(
            [EN_BARE, EN_BARE],
            client=client,
            language="en",
            context=CorrectionContext(),
            checkpoint_dir=tmp_path,
            languages=["zh"],
        )
    assert client.requests == []
