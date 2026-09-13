# PLAN — Make DaVinci-MCP do what it claims (+ OpenAI-compatible providers)

Working plan for the remediation effort identified in the 2026-08 repo scan.
Status tracker at the bottom; tick items as they land. Every phase ends green:
`uv run ruff check . && uv run mypy packages/resolve-mcp/src packages/director/src && uv run pytest`.

## Mission

1. **Truthfulness:** every claim in README.md / DECISIONS.md / tool descriptions must be
   either true after this work, or deleted.
2. **The offline pipeline must actually build a timeline** (today it produces an empty one
   and reports `completed_approved`).
3. **The live path must speak the real DaVinci Resolve scripting API**, be reachable from
   the CLI, and degrade honestly when Resolve is absent.
4. **New feature: provider-agnostic LLM layer.** Gemini becomes one implementation; an
   OpenAI-compatible client (OpenRouter default) unlocks any OpenRouter model.

## Non-negotiable invariants (regressions here are failures)

- Offline mode never needs a key: `--fast`, `gemini=None`/`llm=None`, fake backend, CI.
- Layers communicate over MCP only; do not widen director's undeclared resolve-mcp import.
- Mutating tools return StateDelta snapshots; destructive ops stay double-gated.
- Director verdicts are never force-accepted; `FAILED` is a hard stop.
- Seconds are the wire format at the tool boundary; conversions live in `timecode.py`.
- Tests assert outcomes (timeline contents, empty error lists), not just store rows.

---

## Phase 0 — Guardrails first (make every bug reproducible)

Write the failing tests BEFORE fixing anything. They encode the contracts later phases implement.

- [x] Fix `test_cli_smoke.py` env-var name (`DIRECTOR_RUN_STORE` → `DIRECTOR_RUN_STORE_DIR`) so tests stop writing `./runs/` into the repo root.
- [x] `test_pipeline.py::test_auto_builds_nonempty_timeline_without_errors` — run orchestrator end-to-end on fake backend; assert ≥1 item on the timeline AND `edit_result.errors == []`. **Fails today (empty timeline + frozen-model noise).**
- [x] `test_timecode.py::test_dropframe_roundtrip_sweep` — property sweep `frames_to_timecode → timecode_to_frames` over full 24h at 29.97DF and 59.94DF; assert zero mismatches and no exceptions. **Fails today.**
- [x] `test_live_backend.py::test_calls_only_documented_api` — maintain a literal allow-list of real-API method names (see `.opencode/skills/davinci-mcp-dev/SKILL.md` table); fail if the harness log records anything outside it. Prevents future invented methods.
- [x] `test_client_stdio.py::test_iserror_raises` — stub session returning `isError=True`; assert client raises a typed error instead of returning a payload.
- [x] `test_editor.py::test_symbolic_binding_resolves` — plan with append + fade-on-`<item:0>`; assert fade applied to the appended id.

## Phase 1 — Offline pipeline correctness (critical bug #1)

- [x] **Import media before appending.** Editor gains an ensure-media step: collect `source_path`s from the plan/context, call `import_media`, map contextualizer clip-ids → pool ids, rewrite op args (`editor.py`, new `_ensure_media_imported` next to `_ensure_project_exists`). Keep it idempotent (skip paths already in pool via `list_media_pool`).
- [x] **Stop mutating frozen models.** Replace `call.ok = True/False` assignments (editor.py:157,168) with constructing a new `ToolCallRecord` via `model_copy(update={...})` or building the final record once. Root cause of the error-masking crash.
- [x] **Make symbolic binding real.** Planner emits `"__symbolic_id__": "<item:N>"` inside APPEND args (or editor derives it deterministically per append); `_resolve_args` substitutes `<item:N>` from `item_id_map` everywhere it appears (args dict walk, not just top-level key). Fix `_last_item_id` to select the item actually appended (match by media_clip_id + start_seconds from the delta's `after`, falling back to diffing before/after item-id sets).
- [x] **Propagate execution failure into status.** If `edit_result.errors` non-empty on APPROVED plans, run ends `COMPLETED_WITH_WARNINGS` minimum; if no ops landed, `FAILED` with surfaced errors. Update `_terminal_status` accordingly (pipeline.py).
- [x] Acceptance: Phase-0 regression tests pass; manual `uv run director auto <tmp clips dir> -m x.wav --fast` prints JSON with empty errors and the fake backend holds N items with fades applied.

## Phase 2 — SMPTE drop-frame encoder (critical bug #3)

- [x] Rewrite `_frames_to_dropframe_tc` as the exact inverse of `_dropframe_to_frames` (derive encode from the verified decoder semantics; drop the phantom "binary search" comment). Handle both 29.97 (D=2) and 59.94 (D=4).
- [x] Keep decode strictness (reject forbidden ff values); encoder output must never contain them.
- [x] Acceptance: Phase-0 sweep passes for 24h × both DF rates; SMPTE reference vectors hold (`00:10:00;00 ⇄ 17982`, `00:00:59;59 ⇄ 1797`, hour-boundary cases); nondrop paths unchanged.

## Phase 3 — Live backend vs REAL Resolve API (critical bug #2)

Design decisions locked up front:

- **Track indexing:** wire format becomes **1-based everywhere** (matches Resolve V1/A1 semantics). Fake backend creates tracks index 1(video)/2(audio); update schemas docstrings, planner defaults, editor, tools docs, and tests in one commit. Tool schema gains explicit validation `>= 1`.
- **Stable IDs:** `DaVinciResolveBackend` keeps a session-scoped registry mapping opaque stable ids (`tl_item_<n>`, `clip_<n>`) → last-seen Resolve objects, refreshed by scanning tracks/pool on each state read; match fallback by (name, trackIndex, startFrame) tuple. Ids returned by one hydration remain resolvable in later calls within the server process lifetime. Document cross-process limitation in README.
- **Honest degradation:** every method that cannot honor its contract against live Resolve raises `ResolveUnavailableError`/`InvalidStateError` with a clear message — never a fabricated success.

Tasks:

- [x] Rewrite calls per documented API: `AppendToTimeline([{mediaPoolItem, startFrame, endFrame, mediaType?, trackIndex?}])`; `GetItemListInTrack(type, idx)`; per-key `SetProperty("ZoomX", v)` etc.; `AddMarker(frame, "Blue"-style color, name, note)` with enum-value capitalization map; render via `SetRenderSettings({"TargetDir","CustomName"})` + `AddRenderJob()` + `StartRendering(jobId)` + `GetRenderJobStatus(jobId)`; delete timeline via `MediaPool.DeleteTimelines([tl])` with graceful fallback+error if version lacks it; delete clip via `TimelineItem.Delete()`.
- [x] Re-model `tests/fake_resolve.py` to expose ONLY the documented surface above (names/signatures/return shapes); update `davinci_backend.py` until live-backend tests pass against the re-modeled harness. The harness now mirrors Blackmagic's README, breaking circularity.
- [x] Unify divergent semantics between backends or document them at the tool level: `insert_clip` (decide: implement record-frame placement via `recordFrame`, no silent shift), `import_media` bin support (AddSubFolder + ImportMedia into that folder), `delete_timeline` (real deletion), `restart_app` (explicit "quit only" rename of description), `create_bin` (verify success, don't suppress-and-return).
- [x] `list_projects` uses `GetProjectListInCurrentFolder()`-equivalent documented method instead of only-current-project.
- [ ] Manual smoke checklist (needs real Resolve Studio — owner runs): create project/timeline, import 2 clips, append/move/marker, queue+start render, and confirm the one remaining doc-ambiguous detail: that TimelineItem marker frames are item-relative (read and write use the same convention, so a mismatch shows as a marker in the wrong place, not as an error). The other two are now self-detecting: GetRenderJobStatus keys are matched case-insensitively and a payload with no status field raises, and append_clip reads the created item's length back, so an inclusive clipInfo endFrame reports itself with both frame counts instead of corrupting the cut. Record results in DECISIONS.md. The offline suite cannot cover this.

## Phase 4 — CLI wiring + stdio honesty (high bugs #6/#8)

- [x] `director auto/interactive --backend davinci` actually launches the davinci-backed server through `StdioResolveClient.default` (cli.py currently hard-rejects). Validate flags; keep `fake` default.
- [x] Respect settings when flags omitted: make CLI args `None`-defaulted so `DirectorSettings`/`ResolveMCPSettings` (.env) values apply; document precedence flag > env > default. Same fix server-side so `RESOLVE_MCP_*` vars work via console script (server.py main()).
- [x] `client.call_tool`: check `result.isError` → raise `ToolCallError(text)`; editor records it like any op exception. Removes the silent-failure hole over stdio.
- [x] Remove `http` from `.env.example` transport claim (or implement — out of scope; stdio only).
- [x] Acceptance: `director auto ... --backend davinci` boots server subprocess without Resolve present fails with clean `ResolveUnavailableError` surfaced in run status (not a traceback); with fake backend unchanged behavior.

## Phase 5 — Provider abstraction + OpenAI-compatible (OpenRouter)

Architecture:

```
director/llm/
├── base.py        LLMClient protocol: generate_json(system,user,schema)->T ; analyze_video(clip_path,clip_id,prompt)->PerClipMap ; aclose()
├── gemini.py      thin move/wrap of ingestion/gemini_client.GeminiClient (implements protocol)
├── openai_compat.py  AsyncOpenAI(base_url=DIRECTOR_LLM_BASE_URL) client
└── factory.py     get_llm_client(settings) -> LLMClient | None
```

- [x] Settings additions (`settings.py`): `llm_provider: Literal["gemini","openai_compatible","none"] = "gemini"` (env `DIRECTOR_LLM_PROVIDER`), `llm_base_url: str = "https://openrouter.ai/api/v1"`, `llm_api_key: str|None` (env `DIRECTOR_LLM_API_KEY`); keep legacy `GEMINI_API_KEY` working for gemini provider; accept `OPENROUTER_API_KEY`/`OPENAI_API_KEY` as fallbacks for the compat provider. Model ids reuse existing `DIRECTOR_REASONING_MODEL` / `DIRECTOR_VISION_MODEL` (provider-agnostic strings).
- [x] `openai_compat.generate_json`: chat.completions with `response_format={"type":"json_schema", json_schema…}`; on provider rejection fall back to `{"type":"json_object"}` + pydantic-validate + one repair retry with validator message. Parse errors raise `InvalidModelOutput` like GeminiClient.
- [x] `openai_compat.analyze_video`: no video upload exists in OpenAI-compat APIs — extract up to N=8 evenly-spaced keyframes via `ffmpeg` (already a recommended dep) to temp jpgs, send as base64 `image_url` parts with the prompt; if ffmpeg absent or model returns unusable output → return placeholder PerClipMap (same shape gemini=None produces) and log warning. Document degradation in README.
- [x] Factory wiring: contextualizer/planner/director take `LLMClient` (rename field but keep `gemini=` kwarg alias during transition). `--fast` ⇒ factory returns None ⇒ deterministic offline paths untouched. CLI flag `--llm {gemini,openai_compatible,none}` overriding setting.
- [x] Deps: add `openai>=1.60,<3` to director pyproject. No network in tests: unit-test `openai_compat` against a stubbed AsyncOpenAI transport (respx or hand-stubbed client object); add conftest fixture `stub_llm`.
- [x] Docs: .env.example gets a commented provider block incl. OpenRouter examples (`google/gemini-2.5-pro`, `anthropic/claude-sonnet-4.5`, `meta-llama/…`); README section "Choosing a model provider".
- [x] Acceptance: with `DIRECTOR_LLM_PROVIDER=openai_compatible` + fake stub, full auto pipeline runs offline-green; with provider unset everything behaves exactly as today.

## Phase 6 — Truthfulness sweep + hygiene

- [ ] **LICENSE:** still unchosen — a licence is the owner's call, so nothing has been committed. README now says so explicitly instead of linking a file that does not exist.
- [x] README: fix Quickstart (`uv sync --all-extras --all-packages`), Development section (scoped mypy command), remove/mark unverifiable claims until smoke-tested (live-path feature bullets gated on Phase 3 checklist), document stable-ID limitation + provider selection + honest `insert_clip` semantics.
- [x] DECISIONS.md: correct the harness-circularity claim (harness models documented API now, still no substitute for the manual smoke checklist), drop-frame round-trip claim becomes true again post-Phase 2, resume claim scrubbed or implemented.
- [x] Resume: IMPLEMENTED (owner chose build-over-scrub). `Orchestrator.resume_run` re-executes the last agreed plan (context snapshot persisted per-run); `director resume <run_id>` CLI command; refuses in-flight runs; unknown ids fail cleanly.
- [x] mypy overrides: add `packages/director/tests/__init__.py` (aligns module naming with the `tests.*` override) OR extend override list to `test_*`; then `uv run mypy .` must pass — update README to whichever command is canonical afterwards.
- [x] Migrate `[tool.uv] dev-dependencies` → `[dependency-groups] dev` (kills the deprecation warning on every uv command).
- [x] Dead code purge: `TimeConverter.parse_input` stub, unused `append_counter`, `('FileName' if False else 'File')`, `_wrap_state_delta` identity shim, dead `_HAS_LIBROSA` try-block (make it a real import guard), unreachable `if info is None` in davinci delete_clip, `ClipMood` placeholder (keep only if v2 plans say so).
- [x] `.env.example`: `DIRECTOR_RUN_STORE_DIR`, RESOLVE_SCRIPT_LIB real paths (`founl.dll` / `fusionscript.so` per OS), drop dangling "README Resolve Bootstrap" reference (add that README section instead), PYTHONPATH points at Scripting/Modules dir.
- [x] StubResolveClient.list_tools returns the full registered set (or derives from the dispatcher table) instead of 6 hardcoded names.
- [x] FakeResolveBackend: scope bins/clips/timelines per project; fix duplicate-name disambiguation collision case; set `is_modified` on mutation.
- [x] CI: keep lint→typecheck→test order; add `-W error::DeprecationWarning`? (only if cheap); job stays offline-fast.

---

## Sequencing & effort notes

Phases 0→1→2 are strictly ordered (tests first, then the two critical fixes).
Phase 5 is independent of 3/4 — can proceed anytime after Phase 0.
Phase 3 is the long pole; its manual-smoke step requires the owner's Resolve Studio.
Phase 6 lands last so docs describe finished reality.

## Open questions for the owner (non-blocking)

1. License text: MIT OK? (README already implies open-source.)
2. Minimal `resume` now, or scrub the claim and defer?
3. Track-index convention switch (wire format → 1-based) approved? It touches tool schemas/tests but matches Resolve semantics and kills a whole divergence class.
4. Preferred OpenRouter default model id for docs/examples?

## Progress log

- 2026-08-25: Plan created after full-repo audit. Environment prepared: pylsp+mypy+ruff LSP stack installed and smoke-tested; project opencode.json added (validates against published schema); uv venv relocated off /mnt/c to `~/.venvs/davinci-mcp` (9p was silently corrupting installs); dev-deps extended with python-lsp-server/pylsp-mypy/python-lsp-ruff; all 117 tests green (~4s).
- 2026-08-26: Waves 0-2 of remediation executed. Phase 0 guardrails repaired (sweep-limit bug, sabotage impl, marker semantics); Phase 1 landed (import_media step, frozen-model fix, symbolic binding incl. per-append counters, APPROVED-only execution, execution-aware status mapping); Phase 2 landed (DF encoder rewritten as decoder inverse, bijective over full 24h x both rates, 15.5M-conversion proof). Suite: 142 passed / 0 failed / 0 xfailed. E2E offline run verified: 6/6 tool calls ok, empty errors, 3 items + per-item fades.
- 2026-08-26 (later): Wave 4 landed — librosa tempo-ndarray crash fixed (`_as_scalar` coercion) with click-track regression test; real librosa import guard replaced dead try-block; MCP stdio child env now inherits parent environment (UV_PROJECT_ENVIRONMENT reaches `uv run resolve-mcp`; no more repo-local .venv on /mnt/c).
- 2026-08-26 (cont.): Wave 5 landed — provider-agnostic LLM layer (`director/llm/`: LLMClient protocol, Gemini adapter, OpenAI-compatible adapter w/ json_schema->json_object fallback + one repair retry + ffmpeg keyframe vision degradation, factory). Settings: DIRECTOR_LLM_PROVIDER/BASE_URL/API_KEY (+OPENROUTER_API_KEY/OPENAI_API_KEY fallbacks); CLI --llm flag; --fast still forces offline. Dep: openai>=1.60,<3. Suite 157 green.
- 2026-08-26 (cont.): Wave 5.5 landed — minimal resume implemented and verified live (iterations bump, 12/12 ok calls on resumed run). Suite 163 green.
- 2026-09-12: Full audit + remediation by Claude Opus (parallel agents for the API audit,
  the MCP protocol probe and the claims audit; the rest done in-session).

  **resolve-mcp.** Live backend rewritten against Blackmagic's documented API after an audit
  found 10 of 28 tools defective: `TimelineItem.Delete()` does not exist (now
  `Timeline.DeleteClips`), `AddMarker` needs six arguments, composite modes need the
  `COMPOSITE_*` constants, media ids were `hex(id(proxy))` (now `GetUniqueId()`), frame
  numbers ignored `Timeline.GetStartFrame()`, clip durations were hardcoded to 10s, render
  jobs never set the format/codec, and `add_fade`/`set_speed` wrote invented property keys.
  Unsupported operations are no longer registered at all. structlog was printing to stdout,
  corrupting JSON-RPC for every MCP client. Tool schemas now carry enums and bounds; env vars
  finally apply; `list_projects`/`list_timelines`/`set_current_timeline` added. The fake
  backend became NLE-like (per-project state, real files, ffprobe durations, frame
  quantization, no silent overlaps). The harness was rewritten FROM the documentation and now
  hands out fresh wrappers and starts timelines at frame 86400.

  **director.** The "beat-synced" claim is now true: cuts are beat times (measured: 13/13
  within 0.5 ms on a 117 BPM track), shots tile the requested length, the music is laid on
  A1, and pacing follows the brief. Added plan validation (the same table drives the LLM
  prompt), an outcome-based reviewer, a real feedback loop, media probing, capability-aware
  planning, id remapping, honest statuses, resume-onto-a-new-timeline, and an offline
  instruction parser for interactive mode.

  **LLM.** Gemini structured output was passing `model_json_schema()` (with `$defs`, which
  Gemini rejects) and using uploads before they were ACTIVE; both fixed, SDK moved off the
  0.x preview to 1.x. A live-key smoke script is ready to run.

  Gates: ruff clean, mypy strict clean on 39 files (the blanket `ignore_errors` on
  mcp_client/gemini_client is gone), 290 tests green. Verified end to end against real
  generated media through the actual CLI: auto (with and without music), davinci-without-
  Resolve, no-key, interactive, resume, run list/show.

  **Windows-native verification (2026-09-12).** Everything above was exercised under WSL,
  but Resolve's scripting bridge is an in-process library with no network transport, so the
  server has to run on the Windows install that runs Resolve. Both packages were installed
  into a native Windows CPython 3.12 venv and re-verified there: the bootstrap resolves the
  documented `%PROGRAMDATA%\...\Scripting\Modules` path and survives its absence; the
  server boots on both backends over the real stdio transport; stdout carries JSON-RPC
  frames only (structlog on stderr) under the Windows interpreter too; with Resolve absent
  the davinci backend registers 24 tools, returns the actionable checklist per call and
  stays responsive afterwards; and all 31 advertised tools (fake backend, destructive
  enabled) were called once each over the wire with schema-valid arguments — every one
  reachable, every bound and enum enforced. The director CLI ran natively end to end:
  auto with music (13/13 cuts on beats, 18 ops applied, all reviewer axes 1.0), auto
  against davinci without Resolve (clean failure, zero ops), the no-key path, the
  interactive REPL and resume. README now states the co-location requirement.

  **Adversarial review round (2026-09-12).** An independent reviewer went through the
  director package hunting for ways to falsify its claims, and found real ones. Fixed:
  the interactive path never validated its delta plans (an LLM drifting on an argument
  name got an APPROVED verdict and a raw failure at the Resolve boundary) — it now runs
  the same `validate_plan` gate as auto mode and additionally rejects item ids that are
  not on the live timeline; an interactive session that attempted an edit and applied
  nothing reported `completed_with_warnings` instead of `failed`; the planner's
  short-clip fallback applied a 0.4s floor on top of the clip's real length, planning a
  read past the end of the footage that the validator rejected on every iteration; the
  reviewer scored against the raw number in the brief while the planner caps the cut at
  the music length, so a correct plan could never satisfy it; the offline interpreter
  silently edited clip 1 for "the second clip", read "increase opacity by 20%" as "set
  it to 20%", and let one fade direction's number leak into the other; and the validator
  documented numeric bounds to the model (rotation ±360, zoom multiplier) that it never
  enforced. Each fix landed with a regression test — 313 tests, still ruff- and
  mypy-strict clean.

## Remaining

1. **LICENSE** — owner's decision (see Phase 6).
2. **Manual smoke test on real Resolve Studio hardware** — the only way to close the last
   documentation-verified gap. Everything else is green offline, on WSL and on native
   Windows.
