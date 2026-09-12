# AGENTS.md

Guidance for AI coding agents working in this repo. Design rationale lives in [DECISIONS.md](DECISIONS.md); the remediation roadmap and its remaining items live in [plan.md](plan.md) — check it before starting work here.

## Commands

```bash
uv sync --all-extras --all-packages   # REQUIRED setup (see gotcha below)
uv run ruff check .                   # lint
uv run mypy packages/resolve-mcp/src packages/director/src   # type-check (strict, scoped)
uv run pytest                         # full suite (~290 tests, no Resolve/API key needed)
```

- **Setup gotcha:** plain `uv sync` installs *only* dev tools (pytest/ruff/mypy) because the workspace-root project has no dependencies. Without `--all-packages`, neither `director` nor `resolve_mcp` is importable and every test fails. CI uses the same flags.
- **Mypy gotcha:** `uv run mypy .` also walks `tests/`, which is deliberately not strict-typed. The scoped command above is canonical and is what CI runs.
- **WSL gotcha (this machine):** uv venvs must live off `/mnt/c` — 9p silently drops files during wheel extraction. `~/.bashrc` exports `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/davinci-mcp`; export it before any `uv` command if a fresh shell lacks it.
- Run in CI order: lint → typecheck → test. Single test: `uv run pytest packages/resolve-mcp/tests/test_timecode.py -k name`.

## Architecture

Two-layer uv-workspace monorepo. The layers communicate **only over MCP** — no shared process, no cross-imported internals:

- `packages/resolve-mcp` — MCP **server** (the "hands"). FastMCP exposing typed editing tools over DaVinci Resolve's scripting API. Dependency-light by design (`mcp`, `pydantic`, `structlog`, `pydantic-settings` only).
- `packages/director` — MCP **client** orchestrator (the "brain"): contextualize → librosa beat detection → plan/review loop → execute. Carries all heavy deps.

Key boundaries an agent must not break:

- **Layer isolation is the intent, but not fully true:** `director/mcp_client/client.py` imports `resolve_mcp.tools`/`resources` (function-level, for the in-process stub path) and director's tests use `FakeResolveBackend` directly — yet `director/pyproject.toml` does **not** declare `resolve-mcp` as a dependency. This only works because the uv workspace installs both. Don't widen this undeclared coupling.
- Backends sit behind the `ResolveBackend` protocol: `FakeResolveBackend` (in-memory) and `DaVinciResolveBackend` (SDK imported only in a guarded runtime block). Director talks to the server via `StdioResolveClient`, which prefers `<this interpreter> -m resolve_mcp` and falls back to `uv run`; tests use `StubResolveClient` wrapping the fake backend.
- Every mutating tool returns a **state delta** ("no silent failures") — preserve this contract when adding or changing tools.
- Destructive ops (`quit_app`, `restart_app`, `delete_timeline`, `delete_media`) are double-gated: server flag/env **and** per-call `confirm=true`, enforced in the backends too.

## Conventions that bite if you get them wrong

- **Tool availability is per backend.** `ResolveBackend.unsupported_tools()` drives registration; the live backend hides `add_fade`, `set_speed`, `add_transition` and `restart_app` because Resolve's documented API cannot do them. Never "implement" one of those by writing undocumented property keys — raise `UnsupportedOperationError`.
- **Track indexes are 1-based**, video tracks first then audio (V1 = 1, A1 = 2 on a fresh timeline).
- **Seconds are relative to timeline zero.** Resolve's own frame numbers are absolute and include the timeline's start timecode (`Timeline.GetStartFrame()`, usually 86400). Offset on every read and write.
- **Snap both ends of a span**, not start + duration — see DECISIONS.md; independent rounding creates one-frame overlaps.
- **Transform units are Resolve's**: pan/anchor in pixels from centre, zoom as a multiplier (1.0 = 100%), rotation in degrees, crop in pixels. Marker positions are item-relative.
- **Ids come from Resolve** (`GetUniqueId()`/`GetMediaId()`), never from `id(obj)`: the bridge returns a fresh wrapper on every call.
- **Logs go to stderr.** stdout is the JSON-RPC stream on stdio; a stray print breaks every client.
- Env var names are prefix + UPPER_SNAKE of the **full** settings field: `run_store_dir` → `DIRECTOR_RUN_STORE_DIR`. CLI flags beat env; unset flags fall through to env (`resolve-mcp` flags default to `None` for exactly this reason).

## Testing quirks

- Everything runs offline: `FakeResolveBackend` plus `packages/resolve-mcp/tests/fake_resolve.py`, a harness modelled on Blackmagic's **documented** API — not on our backend. It hands out fresh wrappers, starts timelines at frame 86400, rejects undocumented `SetProperty` keys, and enforces real arities. `test_calls_only_documented_api` fails the build if the backend calls anything outside the documented allow-list. Model new live behaviour against the documentation, then the harness — never the other way round.
- Tests assert **outcomes** (cut points land on beats, items on the timeline, empty error lists), not that a function was called. The suite once stayed green while the whole pipeline produced an empty timeline.
- `asyncio_mode = "auto"`, `--strict-markers --strict-config`; testpaths cover both packages from the repo root.
- The fake backend refuses files that don't exist, so tests must create real (possibly empty) files — the `make_media`/`clips` fixtures do this.
- `ffprobe` is optional; tests that need real media skip when it is absent rather than failing.

## Director specifics

- `--fast` (or `--llm none`) = deterministic planner/reviewer, no network. Any change must keep this path working: losing the API key must never break tests or the offline pipeline.
- The deterministic planner is genuinely beat-synced: cuts are beat times, shots tile the target length, the music goes on A1. If you touch `planner.py`, keep `test_planner_beatsync.py` honest.
- Plans go through `plan_validation.validate_plan` before execution, and `VERB_SPECS` there is also what the LLM prompt shows the model — update both by updating the table.
- Verdicts (`APPROVED` / `ACCEPTED_WITH_WARNINGS` / `FAILED`) are never force-accepted; `FAILED` is a hard stop, and a structurally invalid plan can't be approved.
- A run that applied nothing reports `failed`. Don't "soften" statuses.
- Audio decode goes through `soundfile.read()` directly (not `librosa.load` on paths) — the audioread/aifc path breaks on newer Pythons.
- Style: ruff line-length 120, isort rules enabled (`I`); mypy strict on `src/`, relaxed for tests.
