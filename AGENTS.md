# AGENTS.md

Guidance for AI coding agents working in this repo. Design rationale lives in [DECISIONS.md](DECISIONS.md); active remediation roadmap lives in [plan.md](plan.md) — check it before starting work here.

## Commands

```bash
uv sync --all-extras --all-packages   # REQUIRED setup (see gotcha below)
uv run ruff check .                   # lint
uv run mypy packages/resolve-mcp/src packages/director/src   # type-check (strict)
uv run pytest                         # full suite (~123 tests, <15s, no Resolve/Gemini needed)
```

- **WSL gotcha (this machine):** uv venvs must live off `/mnt/c` — 9p silently drops files during wheel extraction (jedi once extracted as an empty stub). `~/.bashrc` exports `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/davinci-mcp`; if a fresh shell lacks it, export before any `uv` command.
- **Setup gotcha:** plain `uv sync` installs *only* dev tools (pytest/ruff/mypy) because the workspace-root project has no dependencies. Without `--all-packages`, neither `director` nor `resolve_mcp` is importable and every test fails. This matches CI (`ci.yml` uses `uv sync --all-extras --all-packages`).
- **Mypy gotcha:** `uv run mypy .` (as shown in README) fails with errors inside `tests/`; the `tests.*` overrides in `pyproject.toml` only apply with the scoped command above. Use the scoped form.
- Run in CI order: lint → typecheck → test. Single test: `uv run pytest packages/resolve-mcp/tests/test_timecode.py -k name`.

## Architecture

Two-layer uv-workspace monorepo. The layers communicate **only over MCP** — no shared process, no cross-imported internals:

- `packages/resolve-mcp` — MCP **server** (the "hands"). FastMCP exposing 28 typed tools over DaVinci Resolve's scripting API. Dependency-light by design (`mcp`, `pydantic`, `structlog`, `pydantic-settings` only).
- `packages/director` — MCP **client** orchestrator (the "brain"): Gemini contextualize → librosa beat detection → plan/review loop → execute. Carries all heavy deps.

Key boundaries an agent must not break:

- **Layer isolation is the intent, but not currently true:** `director/mcp_client/client.py` imports `resolve_mcp.tools`/`resources` (function-level, for the test stub path) and director's tests use `FakeResolveBackend` directly — yet `director/pyproject.toml` does **not** declare `resolve-mcp` as a dependency. This only works because the uv workspace installs both packages. Don't widen this undeclared coupling.
- Backends are swappable behind the `ResolveBackend` protocol: `FakeResolveBackend` (in-memory) and `DaVinciResolveBackend` (SDK imported only in a guarded runtime block). Director talks to the server via `StdioResolveClient` (spawns `uv run resolve-mcp` with `--uv-project`); tests use `StubResolveClient` wrapping the fake backend.
- Every mutating tool returns a **state delta snapshot** ("no silent failures") — preserve this contract when adding/changing tools.
- Destructive ops (`quit_app`, `restart_app`, `delete_timeline`, `delete_media`) are double-gated: server flag/env `allow_destructive` **and** per-call `confirm=true`.
- Seconds are the wire format at the tool boundary; frame/timecode conversion happens in the backend (`timecode.py` implements SMPTE 12M incl. drop-frame).

## Testing quirks

- Everything runs offline: `FakeResolveBackend` plus a record/replay harness (`packages/resolve-mcp/tests/fake_resolve.py`) that models the `DaVinciResolveScript` call surface — this catches SDK argument/method-name drift in CI without Resolve installed. Model new live-backend behavior against the harness; real-Resolve verification is a manual smoke test.
- `asyncio_mode = "auto"` and `--strict-markers --strict-config` are on; testpaths cover both packages from the repo root.
- Some tests carry `xfail(strict=True, reason="plan.md Phase N…")` markers — known-red-by-design guardrails for not-yet-landed plan.md phases. When you land a phase's fix, **remove the marker too**: strict mode reports XPASS as a hard failure, so a green fix + stale marker still shows red.

## Environment & conventions

- Both packages load cwd-relative `.env` via pydantic-settings with prefixes `DIRECTOR_*` and `RESOLVE_MCP_*` (see `.env.example`). `GEMINI_API_KEY` is used only by director.
- Env var names are prefix + UPPER_SNAKE of the **full** settings field name: `run_store_dir` → `DIRECTOR_RUN_STORE_DIR`, not `DIRECTOR_RUN_STORE`. resolve-mcp CLI flags (incl. defaults) always beat `RESOLVE_MCP_*` env because `main()` passes argparse values into settings explicitly.
- Repo `opencode.json` runs pylsp via `uv run --no-sync pylsp` (mypy + ruff plugins matching CI gates) — it needs the shared uv venv to exist.
- `--fast` CLI flag = deterministic planner/director, no Gemini; default backend is `fake`. Any change to planner/director must keep these offline fallbacks working — loss of the API key must never break tests.
- Director verdicts (`APPROVED` / `ACCEPTED_WITH_WARNINGS` / `FAILED`) must never be silently force-accepted; `FAILED` is a hard stop.
- Audio decode goes through `soundfile.read()` directly (not `librosa.load` on paths) — the audioread/aifc path breaks on newer Pythons.
- Style: ruff line-length 120, isort rules enabled (`I`); mypy strict on `src/` but relaxed for tests and `director.mcp_client.*`, `ingestion.gemini_client`, `cli`.
