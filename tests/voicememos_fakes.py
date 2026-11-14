import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ytverbatim.acquire import voicememos

APPLE_EPOCH = 978307200
ALL_COLUMNS = (
    "ZUNIQUEID",
    "ZENCRYPTEDTITLE",
    "ZCUSTOMLABEL",
    "ZDATE",
    "ZDURATION",
    "ZPATH",
    "ZEVICTIONDATE",
    "ZFOLDER",
)


wal_connections: list[sqlite3.Connection] = []


@dataclass
class Row:
    uid: str
    title: str | None = "Memo"
    label: str | None = None
    unix: float | None = 1_760_000_000
    duration: float | None = 61.5
    path: str | None = None
    evicted: float | None = None
    audio: bytes | None = b"audio"


def make_library(root: Path, rows: Sequence[Row], *, drop: Sequence[str] = (), wal: bool = False) -> Path:
    folder = root / "Recordings"
    folder.mkdir(parents=True)
    columns = [c for c in ALL_COLUMNS if c not in drop]
    connection = sqlite3.connect(folder / voicememos.DATABASE)
    if wal:
        connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"CREATE TABLE {voicememos.TABLE} (Z_PK INTEGER PRIMARY KEY, {', '.join(columns)})")
    for row in rows:
        values = {
            "ZUNIQUEID": row.uid,
            "ZENCRYPTEDTITLE": row.title,
            "ZCUSTOMLABEL": row.label,
            "ZDATE": None if row.unix is None else row.unix - APPLE_EPOCH,
            "ZDURATION": row.duration,
            "ZPATH": row.path,
            "ZEVICTIONDATE": row.evicted,
            "ZFOLDER": None,
        }
        connection.execute(
            f"INSERT INTO {voicememos.TABLE} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
            [values[c] for c in columns],
        )
        if row.path and row.audio is not None:
            (folder / row.path).write_bytes(row.audio)
    connection.commit()
    if wal:
        # Kept open, so the rows stay in the -wal file instead of being checkpointed into the database.
        wal_connections.append(connection)
        assert (folder / f"{voicememos.DATABASE}-wal").stat().st_size > 0
    else:
        connection.close()
    return folder
