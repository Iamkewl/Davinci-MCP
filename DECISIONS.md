# DECISIONS

Non-obvious choices and the reasoning behind them.

## Repo layout: `packages/{resolve-mcp,director}` under a uv workspace

**Why:** clean double-isolation. `resolve-mcp` must stay dependency-light (only `mcp` +
`pydantic` + `structlog` + `pydantic-settings`); `director` carries all the heavy stack
(`google-genai`, `openai`, `librosa`, `soundfile`, `numpy`, `typer`). A uv workspace lets each
package declare its own deps and gives us one lockfile + one `pytest` run.

**Rejected:** a single flat package with optional extras — extras make the light package
uninstallable cleanly, and we want to guarantee resolve-mcp NEVER imports a model SDK.

## The offline harness is modelled from Blackmagic's docs, not from our own code

**Why:** the first version of `tests/fake_resolve.py` was written by reading
`davinci_backend.py` and modelling whatever it called. That is circular, and it hid a real
bug for months: the harness implemented `TimelineItem.Delete()`, a method Resolve does not
have, so `delete_clip` passed CI and would have raised `AttributeError` against Resolve.

The harness is now written from the documented scripting API (the sourced inventory lives in
the audit notes), and deliberately reproduces the parts of Resolve that catch mistakes:

* every accessor returns a **fresh wrapper object**, like Resolve's Python bridge, so code
  that keys anything on `id(obj)` fails here exactly as it would live (the old backend keyed
  the media pool on `hex(id(...))`);
* timelines start at frame **86400** (01:00:00:00), so forgetting `GetStartFrame()` is an
  hour-sized error, not a silent one;
* only documented methods exist, `SetProperty` refuses undocumented keys, and `AddMarker`
  requires its full argument list.

`test_calls_only_documented_api` then fails the build if the backend calls anything outside
an allow-list derived from that documentation. This is still not a substitute for the manual
smoke test on real hardware — it is a substitute for *inventing* an API.

**Rejected:** mocking `DaVinciResolveScript` with `unittest.mock` — a mock accepts every
method name, which is precisely the failure mode we needed to catch.

## Capability-aware tool registration

**Why:** Resolve's documented API cannot set fade handles, retime a clip, or create a
transition. The previous build either faked these (writing invented property keys) or failed
at call time. Now the backend declares `unsupported_tools()` and the server simply does not
register them, so `list_tools` tells an MCP client the truth up front and a planner can only
choose operations that exist. The methods still exist on the backend and raise
`UnsupportedOperationError` if called directly.

**Consequence:** the tool count is per-backend (fake: 27, +4 destructive; davinci: 24, +3),
which is why the README no longer advertises a single number.

## Honest degradation over fabricated success

**Why:** silent success is the worst failure mode in an automation tool — you only discover
it when you open the timeline. Concretely: `move_clip` has no API, so it deletes and
re-appends the item and *reports the new id* via `StateDelta.id_remap` instead of pretending
the id survived; `add_render_job` fails loudly if the requested format/codec isn't offered by
this Resolve build; a run that applied zero operations reports `failed`, never "completed".

## Time model: seconds on the wire, frames underneath

**Why:** Resolve mixes frame counts (positions), seconds (durations), and timecodes (SMPTE).
Tools accept **seconds relative to the start of the timeline**; `resolve_mcp.timecode`
converts, including true SMPTE 12M drop-frame for 29.97/59.94 (verified round-trip over a
full 24 hours). Both backends quantize to whole frames and report the snapped value back,
because that is what an NLE does and the caller needs to see it.

A subtlety worth recording: a clip's span is snapped by rounding **both ends** rather than
the start plus the duration. Rounding independently can push a shot one frame into the next
one, so two shots that tile perfectly in seconds collide on the frame grid — which Resolve
then refuses.

## Tool surface: per-operation, type-hinted, self-describing

**Why:** the prior build's `timeline(action="append_clip", ...)` dispatch tool caused LLMs to
drop/flip arguments. One function per operation, typed with the real enums and bounded
`Annotated` types, so the generated JSON schema advertises legal values (`"enum": [...]`,
`minimum`) instead of a bare string. A client should not have to guess a marker colour.

## State delta returns on mutations

**Why:** every mutating tool returns the changed state snapshot, and the live backend reads
properties back out of Resolve to build it — so the delta is evidence the edit landed, not a
restatement of the request. Resources expose read-only state at any time via
`resolve://project` / `resolve://media-pool` / `resolve://timeline/current`.

## Logs go to stderr, always

**Why:** on the stdio transport stdout carries the JSON-RPC frames. structlog's default
factory prints to stdout, which corrupted the stream for every MCP client before an
`initialize` response could be parsed. A regression test asserts stdout stays empty.

## Destructive gate: `--allow-destructive` flag + per-call `confirm=true`

**Why:** two layers, both intentional. The flag prevents accidental invariant violations
during testing and CI; the per-call `confirm=true` prevents single-shot accidents even when
the flag is on. Both backends enforce the flag internally too, so the gate does not depend on
the server layer alone. Destructive list: `quit_app`, `restart_app`, `delete_timeline`,
`delete_media`.

## Plans are validated before they run

**Why:** a plan is a list of tool calls with free-form argument dicts, so a model can produce
something plausible that fails halfway and leaves a half-built timeline. `plan_validation`
checks argument names/types/units, 1-based track indexes, source ranges against real clip
durations, dangling `<item:N>` references, tools this backend lacks, and overlapping shots.
The same `VERB_SPECS` table generates the prompt text that tells a model what the arguments
are, so the validator and the documentation cannot drift apart.

## The reviewer scores geometry, not vibes

**Why:** the previous scorer computed `appends / beat_count`, which made a good 3-shot plan
against a 60-beat track look like a 5% failure while never checking a single cut point. The
axes now measure the plan: how many cuts land within a frame of a detected beat, whether the
timeline is covered without gaps, whether it is structurally valid, whether the length
matches the brief, whether the clips vary, and whether anything destructive appears.

Verdicts stay honest: `FAILED` is never executed, and a plan with structural problems can
never be `APPROVED` — including when an LLM reviewer liked it. `ACCEPTED_WITH_WARNINGS`
means "acceptable", so once the planner stops improving the plan it *is* built, and the
warnings travel with the run status.

## Feedback actually loops

**Why:** "keep iterating so the planner can act on the warnings" was previously untrue — the
planner received the same request every time. The reviewer's issues and the validator's
problems are now part of the next `PlannerRequest`, both for the deterministic planner (which
tightens beat snapping or pacing) and for the LLM (which is shown its rejected plan and the
specific problems). The loop also stops as soon as a plan repeats.

## Resume rebuilds onto a new timeline

**Why:** replaying a plan into the timeline it already built collides with the existing
items. Resume creates `"<timeline> (resume N)"`, so the original cut — and any manual work on
it — survives.

## Run store: SQLite + JSONL

**Why:** SQLite for indexed query (verdicts, tool calls per run) and JSONL for streaming
(parse, partial read, replay). `run_id` is the join key. Checkpoints after every verdict let
`director resume <run_id>` rebuild the last agreed state.

## Audio decode: librosa + soundfile (not audioread)

**Why:** the deprecated `audioread`/`aifc` path breaks on newer Pythons. We force the
`soundfile` decode backend by bypassing `librosa.load(...)` for path inputs and decoding via
`soundfile.read()` directly, then handing the numpy array to `librosa` for analysis.

## Media is probed, not guessed

**Why:** the planner cannot cut to length without knowing how long a clip is, and a
hardcoded default (the old fake backend claimed every clip was 10 seconds) produces plans
that read past the end of the footage. `ffprobe` is used when present; when it isn't, the
duration is reported as `0.0` meaning *unknown*, and both the planner and the editor treat it
as "assume it fits, clamp at execution time" rather than inventing a number.

## Offline planner + reviewer fallbacks

**Why:** both paths produce deterministic results when no model is configured, so CI, `--fast`
and users without a key get a real, working pipeline — the offline planner is genuinely
beat-synced, not a placeholder. The LLM-driven paths go through the same validation and
scoring; only the source of the plan changes. Loss of the API key never blocks the build,
the tests, or a usable edit.
