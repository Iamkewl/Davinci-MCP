# DECISIONS

Non-obvious choices and the reasoning behind them.

## Repo layout: `packages/{resolve-mcp,director}` under a uv workspace

**Why:** clean double-isolation. `resolve-mcp` must stay dependency-light (only `mcp` +
`pydantic` + `structlog` + `pydantic-settings`); `director` carries all the heavy stack
(`google-genai`, `librosa`, `soundfile`, `numpy`, `typer`). A uv workspace lets each package
declare its own deps and gives us one lockfile + one `pytest` run.

**Rejected:** a single flat package with optional extras — extras make the light package
uninstallable cleanly, and we want to guarantee resolve-mcp NEVER imports Gemini.

## Backend abstraction: `ResolveBackend` protocol + `FakeResolveBackend` + `FakeResolve` harness

**Why:** the agent cannot run DaVinci Resolve in CI, on a contributor laptop without
Studio installed, or in a deterministic unit test. We use two complementary things:

1. `FakeResolveBackend` (in-memory model of project, media, timeline items, effects,
   render queue) — the unit-test surface for everything.
2. `tests/fake_resolve.py` — a record/replay harness modelling the subset of
   `DaVinciResolveScript` that the live backend talks to. The harness records every
   Resolve-scripting-API call (method name + args) into a `CallLog`; the
   `DaVinciResolveBackend` constructs against this fake when the SDK is absent, so we
   catch version drift, argument-name typos, and method-name typos in CI without
   needing a real Resolve.

`DaVinciResolveBackend` only imports the SDK at runtime inside a guarded block; the
real backend's full Live surface (Phase-1+2 items, render jobs, destructive gates) is
exercised through the harness on every CI run, then verified once by a human against
the actual Resolve UI per the manual smoke test.

**Rejected:** a mock library (`unittest.mock`) wrapping the live scripting module — the
real module isn't importable in CI, so mocking the import name doesn't exercise anything
real-shape. The fake backend models explicit data; the fake harness models the live
method-call surface. Combination gives us high coverage on both contracts.

## Time model: one converter, three input forms

**Why:** Resolve mixes frame counts (positions), seconds (durations), and timecodes
(SMPTE). Tools accept seconds as the wire-format in Phase 1 (no time/timecode input parsing
at tool boundary yet — the *backend* is the single source of truth). The :mod:`resolve_mcp.timecode`
module fully supports the SMPTE 12M encoder / decoder for non-drop and drop-frame rates; tested
exhaustively round-trip and against the SMPTE 12M reference values (e.g., `00:10:00;00 → 17982`
for 29.97 df). Float seconds → frames round-trip is bounded by one frame for NTSC fps ratios
(sub-frame loss is unavoidable via IEEE-754; the API documents this).

## Tool surface: per-operation, type-hinted

**Why:** the prior build's `timeline(action="append_clip", ...)` dispatch tool caused
LLMs to drop/flip arguments. FastMCP generates JSON schemas from Python type hints, so
typing every parameter cleans the schemas. We expose one function per operation and never
string-dispatch inside a tool. 28 individual tools.

## State delta returns on mutations

**Why:** silent success is the worst failure mode. Every mutating tool returns the changed
state snapshot; the editor reads state back after each tool call (the "no silent failures"
contract). Resources expose read-only state at any time via `resolve://project` /
`resolve://media-pool` / `resolve://timeline/current`.

## Destructive gate: `--allow-destructive` flag + per-call `confirm=true`

**Why:** two layers, both intentional. The flag prevents accidental invariant violations
during testing and CI; the per-call `confirm=true` prevents single-shot accidents even
when the flag is on. Destructive list:
`quit_app`, `restart_app`, `delete_timeline`, `delete_media`.

## Director verdict: never silently force-accept

**Why:** the prior build force-accepted a low-quality plan on max iterations. Director
returns an explicit terminal verdict: `APPROVED`, `ACCEPTED_WITH_WARNINGS`, or `FAILED`.
The editor treats `FAILED` as a hard stop. Loop budget is configurable
(`DIRECTOR_MAX_PLANNER_ITERATIONS`, default 5). When the loop exhausts without APPROVED the
run status is `FAILED` and the latest verdict is surfaced.

## Run store: SQLite + JSONL

**Why:** SQLite for indexed query (verdicts, tool calls per run) and JSONL for streaming
(parse, partial read, replay). `run_id` is the join key. Checkpoint after every Director
verdict so a potential `--resume <run_id>` lands at the last agreed state, not the middle
of an edit.

## Audio decode: librosa + soundfile (not audioread)

**Why:** the deprecated `audioread`/`aifc` path breaks on Python 3.13. We force the
`soundfile` decode backend by bypassing `librosa.load(..)` for path inputs and decoding via
`soundfile.read()` directly, then handing the numpy array to `librosa` for analysis.
Decoded audio is soundfile-backed end-to-end.

## Offline planner + director fallbacks

**Why:** both `Planner.run` and `Director.run` produce deterministic defaults when
`gemini=None`. This means:
* CI / `directed tests run without a Gemini API key.
* `--fast` in the CLI is a real, working path for users without a key.
* Production swap-in is one constructor change.

The LLM-driven paths still go through the same code — only the inputs change. Loss of
the API key never blocks the build / test / fast mode.
