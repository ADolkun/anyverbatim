import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from anyverbatim.contract import ContextDocument, dump_document
from anyverbatim.index import INDEX_MD, write_index


def recording(root: Path, digit: str, title: str, recorded_at: datetime | None, duration_s: float) -> None:
    sha = digit * 64
    document = ContextDocument.model_validate(
        {
            "schema_version": "2.0",
            "generator": {"name": "anyverbatim", "version": "0.0.1"},
            "created_at": "2026-01-02T03:04:05+00:00",
            "source": {
                "kind": "recording",
                "id": f"rec-{sha[:16]}",
                "audio_sha256": sha,
                "title": title,
                "file_name": "example.m4a",
                "origin": "file",
                "recorded_at": recorded_at,
                "duration_s": duration_s,
                "language": None,
            },
            "transcript": {
                "language": "en",
                "languages": ["en"],
                "script": "n/a",
                "runs": [],
                "punctuation": "engine",
                "correction_llm": None,
                "diarization": None,
            },
            "speakers": [],
            "chapters": [],
            "paragraphs": [],
            "segments": [],
            "quality": {
                "speech_seconds": 0,
                "transcribed_seconds": 0,
                "decoder_anomalies": 0,
                "flags_by_kind": {},
                "warnings": [],
            },
            "llm_usage": [],
        }
    )
    folder = root / f"rec-{sha[:16]}"
    folder.mkdir(parents=True)
    (folder / "context.json").write_text(dump_document(document), encoding="utf-8")


def test_the_index_lists_recordings_newest_first_with_undated_last(tmp_path: Path) -> None:
    recording(tmp_path, "a", "Older | call", datetime(2026, 1, 2, 9, 30, tzinfo=UTC), 65.0)
    recording(tmp_path, "b", "Undated memo", None, 3725.0)
    recording(tmp_path, "c", "Newest [sync]", datetime(2026, 11, 3, 14, 5, tzinfo=timezone(timedelta(hours=8))), 59.9)
    (tmp_path / "rec-broken").mkdir()
    (tmp_path / "rec-broken" / "context.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rec-dddddddddddddddd").mkdir()
    (tmp_path / "rec-dddddddddddddddd" / "context.json").write_text(json.dumps({"bad": 1}), encoding="utf-8")

    path = write_index(tmp_path)
    assert path == tmp_path / INDEX_MD
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("| ") and "---" not in line][1:]
    assert rows[0].startswith("| 2026-11-03 14:05 | [Newest \\[sync\\]](rec-cccccccccccccccc/context.md) | 00:59 |")
    assert rows[1].startswith("| 2026-01-02 09:30 | [Older \\| call](rec-aaaaaaaaaaaaaaaa/context.md) | 01:05 |")
    assert rows[2].startswith("| - | [Undated memo](rec-bbbbbbbbbbbbbbbb/context.md) | 1:02:05 |")
    assert len(rows) == 3, "unreadable documents are left out"


def test_the_index_is_regenerated_from_what_is_present(tmp_path: Path) -> None:
    recording(tmp_path, "a", "First", None, 1.0)
    write_index(tmp_path)
    recording(tmp_path, "b", "Second", datetime(2026, 1, 1, tzinfo=UTC), 1.0)
    text = write_index(tmp_path).read_text(encoding="utf-8")
    assert text.index("Second") < text.index("First")


def test_an_empty_root_says_so(tmp_path: Path) -> None:
    assert "No recordings yet." in write_index(tmp_path / "recordings").read_text(encoding="utf-8")


def test_a_document_that_is_not_utf8_is_left_out(tmp_path: Path) -> None:
    recording(tmp_path, "a", "Kept", None, 1.0)
    (tmp_path / "rec-eeeeeeeeeeeeeeee").mkdir()
    (tmp_path / "rec-eeeeeeeeeeeeeeee" / "context.json").write_bytes(b"\xff\xfe not utf-8")
    text = write_index(tmp_path).read_text(encoding="utf-8")
    assert "Kept" in text
    assert "eeeeeeeeeeeeeeee" not in text
