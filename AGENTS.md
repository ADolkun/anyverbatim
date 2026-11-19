# AGENTS.md — anyverbatim
Lean contract for coding agents. Keep it short; link to docs for detail.

## Purpose
`anyverbatim` turns YouTube videos and the user's own recordings (audio and video files, iPhone Voice Memos) into verified, timestamped transcripts for LLM context. It uses the user's own browser login for Premium and members-only videos, runs ASR locally on Apple Silicon (MLX), and focuses on Mandarin and English. v1 is transcript-only. It is published publicly and consumed as a library by other applications.

## Principles
- **Accuracy over speed.** Never trade transcript quality for runtime.
- **Fail loud.** Never silently fall back to a weaker path or to an anonymous session; raise a clear, actionable error.
- **Local-first.** ASR runs locally. Cookies never leave the machine and are never logged or printed.
- **Resume-safe and idempotent.** Reuse verified artifacts and write them atomically (temp file plus replace).

## Privacy (public repo)
- Never commit personal data, absolute home paths, cookies, media, transcripts, or the IDs or titles of members-only videos.
- Eval data lives outside the repo at `ANYVERBATIM_EVAL_DIR` (default `~/.local/share/anyverbatim/eval`).
- Run `make install-hooks` once per clone. The pre-commit hook and `make privacy-check` run `scripts/check_private.py`, which also reads extra local patterns from the untracked `.git/info/private-patterns`.
- Use obviously fake fixtures and placeholder paths.

## Toolchain
- Use `uv`; run everything as `uv run ...`. Do not assume a bare `python`.
- Python 3.14. Ruff checks every rule (each opt-out in `pyproject.toml` says why) and formats; pyrefly runs strict on source and tests, and its warnings fail too. Pytest runs in strict mode with warnings as errors.
- Fix a finding rather than silence it. When a suppression is right, name the code (`# noqa: CODE`, `# pyrefly: ignore[kind]`) and say why if it isn't obvious.
- macOS has no `timeout` or `gtimeout`; do not use them.
- Source lives in `src/anyverbatim/`; the public API must stay typed (`py.typed`).

## Verification
`make check` must pass before a task is done: ruff lint, ruff format check, pyrefly strict with zero errors, pytest, and the privacy check. Run the narrowest checks while iterating.

## Conventions
- Conventional commits. No AI attribution and no `Co-Authored-By` trailers.
- Comments explain why, never what.
- The public API is `anyverbatim.__all__` plus `anyverbatim.errors|contract|config` ([docs/library.md](docs/library.md)); `tests/test_public_api.py` pins it, so change it deliberately and list a break in the docs.
- Tests are offline: stub yt-dlp, network, and models. Use temporary directories.

## Shared checkouts
Several agents may share one checkout. Never run `git reset`, `git stash`, `git checkout --`, `git restore`, or `git clean`; they silently destroy other agents' uncommitted work. Read an earlier version with `git show <rev>:<path>`. Edits in files outside your assignment belong to someone else: leave them and scope your checks to your own files.
