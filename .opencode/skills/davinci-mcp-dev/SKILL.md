---
name: davinci-mcp-dev
description: Use when working in the DaVinci-MCP repo (packages/resolve-mcp or packages/director) — covers the REAL DaVinci Resolve scripting API surface vs the fake harness, offline-testing rules, env-var truth table, frozen-pydantic pitfalls, and WSL /mnt/c setup gotchas. Trigger on resolve-mcp, director, ResolveBackend, drop-frame timecode, MCP tools, or any fix/feature work here.
---

# DaVinci-MCP development

Hard-won context for this repo. Read before changing backend/tool/pipeline code.

## The single biggest trap: the record/replay harness is NOT the Resolve API

`packages/resolve-mcp/tests/fake_resolve.py` models whatever method names the
live backend happens to call. It is **circular**: passing `test_live_backend.py`
proves nothing about real Resolve. Several calls historically used methods that
do not exist in Blackmagic's documented scripting API.

Verified-real surface (per official Resolve Scripting API README, v18+):

| Real API | Notes |
| --- | --- |
| `MediaPool.AppendToTimeline([{clipInfo}])` | clipInfo keys: `mediaPoolItem`, `startFrame`, `endFrame`, optional `mediaType` (1=video, 2=audio), `trackIndex`, `recordFrame`. Returns `[TimelineItem]`. |
| `MediaPool.CreateEmptyTimeline(name)` | returns Timeline |
| `MediaPool.ImportMedia([paths])` | returns `[MediaPoolItem]` |
| `MediaPool.DeleteTimelines([timeline])` | exists in 18.x+; prefer over clip-emptying hacks |
| `Timeline.GetItemListInTrack(trackType, index)` | **two args required**; trackType "video"/"audio", index is 1-based |
| `TimelineItem.SetProperty(key, value)` | ONE key per call: "Pan","ZoomX","ZoomY","RotationAngle","CropLeft"... (no "Transform" sub-dict form) |
| `TimelineItem.AddMarker(frameId, color, name, note)` | color is Capitalized ("Blue"), not lowercase |
| `TimelineItem.GetProperty(key)` / `GetProperty()` | |
| `Project.GetRenderJobList()` | list of dicts with "JobId" |
| `Project.GetRenderJobStatus(jobId)` | returns {status info}; NOT GetRenderJob(id) |
| `Project.StartRendering(jobId,...)` / `StartRendering([jobIds])` | takes id STRINGS, never job objects |
| `Project.SetRenderSettings({settings})` | keys incl "TargetDir","CustomName","MarkIn","MarkOut" |
| `ProjectManager.CreateProject/LoadProject/GetCurrentProject` | |

Never invent method names. If unsure, check Blackmagic's README.txt (shipped
with Resolve under Developer/Scripting) or the extremraym.com mirror before
writing a live-backend call.

## Offline-first rules (non-negotiable)

- `--fast` / gemini=None paths MUST keep working; loss of any API key must
  never break tests or the offline pipeline.
- Tests must assert OUTCOMES (items on timeline, empty error lists), not just
  plans/verdicts/store rows. The suite once stayed green while the whole
  auto-pipeline produced an empty timeline.
- Model new live-backend behavior in the harness ONLY using real-API method
  names from the table above.

## Pydantic pitfall that bit us here

`StrictModel` sets `frozen=True`. Assigning `record.ok = True` on such models
raises `frozen_instance` at runtime — this once masked every tool-call error in
editor.py. Never mutate frozen models; use `model_copy(update={...})`.

## Env-var truth table (pydantic-settings prefixes)

Settings field `run_store_dir` → env `DIRECTOR_RUN_STORE_DIR` (NOT
`DIRECTOR_RUN_STORE`; .env.example/tests were wrong once). Same rule for all:
prefix + UPPER_SNAKE field name. resolve-mcp CLI flags always override
RESOLVE_MCP_* env vars because main() passes argparse values explicitly.

## Layout & boundaries

- Layers talk over MCP by design; today `director/mcp_client/client.py`
  imports `resolve_mcp.tools/resources` (function-level, stub path) and
  director tests import FakeResolveBackend directly. director does NOT declare
  resolve-mcp as a dep — do not widen the coupling.
- Wire format at tool boundary = seconds + track index; frame/timecode math
  lives in `resolve_mcp.timecode` (SMPTE 12M; drop-frame encode had an
  off-by-dropped-frames bug — property-test round-trips over full 24h).
- Every mutating tool returns a StateDelta snapshot; destructive ops need
  server flag AND confirm=true.

## Environment (this machine)

- WSL on /mnt/c: uv venvs MUST live off the Windows mount (`9p` silently drops
  files during wheel extraction — jedi once extracted as an empty namespace
  dir). `~/.bashrc` exports `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/davinci-mcp`.
- Setup: `uv sync --all-extras --all-packages` (plain `uv sync` installs dev
  tools only). Verify: `uv run ruff check . && uv run mypy
  packages/resolve-mcp/src packages/director/src && uv run pytest`.
- LSP for opencode: project opencode.json runs `uv run --no-sync pylsp` with
  pylsp-mypy + python-lsp-ruff (matches CI gates exactly).
