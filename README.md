# DaVinci-MCP

[![ci](https://github.com/Iamkewl/Davinci-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/Iamkewl/Davinci-MCP/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

**Hand your clips and a music track to an agent, and let it cut the video for you.**

DaVinci-MCP is an open-source system that automates video editing inside [DaVinci Resolve Studio](https://www.blackmagicdesign.com/products/davinciresolve). It analyzes your raw footage, finds the beats in your soundtrack, plans a beat-synced timeline, and builds it for you — driving Resolve through the [Model Context Protocol](https://modelcontextprotocol.io) (MCP).

---

## Why this exists

This started as pure frustration.

I had a video assignment to turn in for university, and I spent hours doing the part of editing nobody enjoys — scrubbing through clips, hunting for usable moments, chopping them to the beat, nudging cuts a few frames at a time. It's tedious, repetitive work, and it's the same grind whether you're a student, a creator, or anyone who just wants a watchable cut without living inside a timeline.

So I built this for everyone who's been through that same pain. The idea is simple: you shouldn't have to do the mechanical parts by hand. Point it at a folder of clips and a track, tell it the vibe you want, and let the agents handle the busywork — while you stay in control of the result.

---

## What it is

DaVinci-MCP is a **two-layer system**, split cleanly so each half does one job well. The two layers talk to each other **only** over MCP — no shared process, no shared globals, no cross-imported internals.

```
  ┌─────────────────────────────┐        ┌──────────────────────────────┐
  │           director          │  MCP   │          resolve-mcp         │
  │  (the "brain" — MCP client) │ ─────► │   (the "hands" — MCP server) │
  │                             │ stdio  │                              │
  │  • vision analysis of clips │        │  • typed editing tools       │
  │  • librosa beat detection   │        │  • drives Resolve's API      │
  │  • plan → review → execute  │        │  • state-delta verification  │
  │  • SQLite + JSONL run store │        │  • fake backend for testing  │
  └─────────────────────────────┘        └──────────────────────────────┘
```

### Layer 1 — `resolve-mcp` (the hands)

A first-party MCP **server** that exposes DaVinci Resolve's scripting API as individual, type-hinted tools — one function per operation (`append_clip`, `set_transform`, `add_marker`, `add_render_job`, …) instead of a handful of overloaded string-dispatch tools. Every state-changing tool returns a snapshot of what changed, so the caller can *verify* an edit landed rather than hoping it did.

**The tool list reflects what the backend can actually do.** Resolve's documented scripting API has no entry point for fades, retimes or transitions, so with `--backend davinci` those tools are not advertised at all: a client sees 24 tools (27 with `--allow-destructive`) instead of the fake backend's 27 (31 with destructive ops). Nothing pretends to succeed.

It ships with a **`FakeResolveBackend`** that models project/timeline state in memory — so you (and CI) can run and test the whole thing without DaVinci Resolve installed. It behaves like an NLE, not a stub: state is scoped per project, positions snap to the frame grid, media is probed with `ffprobe`, files that don't exist can't be imported, and clips can't silently overlap.

### Layer 2 — `director` (the brain)

An MCP **client** and agent orchestrator that runs the creative pipeline:

1. **Contextualize** — every clip is probed for its real duration/frame rate, and (with an API key) a vision model describes what's in it.
2. **Listen** — `librosa` detects tempo, beats, and onsets in your music.
3. **Plan → Review** — a planner drafts a beat-synced timeline; a reviewer scores what the plan actually does — where each cut lands relative to the beats, whether the timeline is covered, whether it fits the brief — and returns `APPROVED`, `ACCEPTED_WITH_WARNINGS` or `FAILED`. Its issues are fed back into the next planning pass.
4. **Execute** — the plan is checked for structural problems (unknown arguments, out-of-range source ranges, overlapping shots) *before* anything is executed, then built one verified edit at a time.

Every run is recorded to a **SQLite + JSONL run store** so you can inspect exactly what happened — and rebuild it later.

---

## Features

- 🎬 **Clips + music → finished timeline**, automatically.
- 🥁 **Genuinely beat-synced** — cuts land on detected beats (sub-frame), the music is laid under the picture, and shot length follows the brief's energy.
- 🧩 **Typed MCP tools** covering projects, media, timelines, effects, and rendering, with enums and bounds in the schema so any MCP client can discover legal values.
- ✅ **Verified edits** — mutations return before/after state deltas; no silent failures.
- 🔎 **Honest reviews** — the reviewer scores the plan's real geometry, and a structurally invalid plan can never be approved.
- 🙅 **Honest limits** — what Resolve's API cannot do isn't offered (see [Limitations](#limitations)).
- 💾 **Inspectable runs** — full history in SQLite + JSONL; `resume` rebuilds an agreed cut on its own timeline.
- 🛡️ **Safe by default** — destructive ops need a server flag *and* a per-call confirmation.
- 🧪 **Runs without Resolve** — fake backend + a fully offline `--fast` mode.
- 💬 **Interactive mode** — refine the cut conversationally; offline it understands a fixed set of edit instructions and says so plainly when it doesn't understand.

---

## Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (workspace & dependency manager)
- **DaVinci Resolve Studio 18.5+** — *for real editing.* Studio only: the free edition has no external scripting. Enable it under **Preferences → General → External scripting using = Local**.
- **An API key** — for vision + LLM planning (Gemini by default, or any OpenAI-compatible endpoint). Not needed in `--fast`/offline mode.
- **FFmpeg** (`ffprobe`) on your PATH — used to read clip durations and frame rates. Without it, lengths are unknown and the planner is more conservative.

> **Run it on the machine Resolve runs on.** Resolve's scripting bridge is a local, in-process
> library with no network transport, so `resolve-mcp` — and therefore `director`, which launches it
> as a subprocess — must run under an interpreter on the same OS install as Resolve. On Windows that
> means native Windows Python: a WSL interpreter cannot reach a Windows Resolve. The `fake` backend
> has no such constraint.

> You can try everything below **without** Resolve and **without** an API key using the `fake` backend and `--fast` mode.

---

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/Iamkewl/Davinci-MCP.git
cd Davinci-MCP

# 2. Install the workspace (--all-packages is required: a plain `uv sync`
#    installs only the dev tools and neither package is importable)
uv sync --all-extras --all-packages

# 3. (Optional) add your API key for the full pipeline
cp .env.example .env
#   then set GEMINI_API_KEY=... in .env

# 4. Try it end-to-end with NO Resolve and NO API key:
uv run director auto ./clips --music ./music.mp3 \
    --prompt "high-energy 30s reel" --fast
```

`--fast` uses the deterministic planner/reviewer (no model) and the default `fake` backend (no Resolve), so it's the ideal way to see the flow before wiring up the real app.

---

## Usage

### Auto mode — clips + music → timeline

```bash
# Offline dry run (fake backend, deterministic planning)
uv run director auto ./clips -m ./music.mp3 -p "moody cinematic edit" --fast

# The real thing: drive a running DaVinci Resolve Studio
uv run director auto ./clips -m ./music.mp3 -p "moody cinematic edit" --backend davinci
```

| Option | Meaning |
| --- | --- |
| `clips_dir` | Directory of source clips (positional). Non-media files are skipped. |
| `--music`, `-m` | Music track to sync to |
| `--prompt`, `-p` | Your brief (default: `high-energy 30s reel`). A length like "30s" or "1 minute" is honoured. |
| `--backend` | `fake` (default) or `davinci` |
| `--project` | Resolve project to build in (default: the open one, else `auto-reel`) |
| `--timeline` | Timeline name (default: a fresh `Reel <run id>`, so re-running never appends into an existing cut) |
| `--fps` | Timeline frame rate (default: inferred from the clips) |
| `--fast` | Skip the model entirely; deterministic planner + reviewer |
| `--llm` | Provider override: `gemini`, `openai_compatible` or `none` |
| `--uv-project` | Launch the server via `uv run --project DIR` instead of this interpreter |

The command prints a JSON report: run id, status, the timeline it built (name, item count, duration), the reviewer's verdict with per-axis scores, how many operations were applied, and any errors or warnings.

### Choosing a model provider

The planner/reviewer/vision stack is provider-agnostic. Default is Gemini; any OpenAI-compatible endpoint (OpenRouter, vLLM, …) works too:

```bash
# .env
DIRECTOR_LLM_PROVIDER=openai_compatible
DIRECTOR_LLM_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-...
DIRECTOR_REASONING_MODEL=anthropic/claude-sonnet-4.6
DIRECTOR_VISION_MODEL=google/gemini-3-flash
```

Per-run override: `--llm {gemini,openai_compatible,none}`. `--fast` always runs fully offline. OpenAI-compatible endpoints have no video upload, so clips are sampled into keyframes with ffmpeg for the vision pass; without ffmpeg that step degrades to "no visual description" rather than failing the run.

### Interactive mode — refine conversationally

```bash
# rebuild a previous run's cut on a fresh timeline, then refine it
uv run director interactive --fast --from-run <run_id>

# or refine whatever is open in Resolve
uv run director interactive --backend davinci
```

With a model configured, instructions are free-form. Offline (`--fast`) it understands a fixed vocabulary — type `help` to see it:

```
fade in the first clip 1s     slow clip 2 to 0.5x        zoom clip 2 to 120%
set opacity of clip 1 to 60%  mark clip 1 'hook' at 0.5s move clip 3 to 12s
rotate clip 1 by 5            blend mode screen on clip 2  delete clip 4
```

Anything it can't parse is reported as such — it never invents an edit you didn't ask for.

### Inspect and rebuild runs

```bash
uv run director run list           # every run in the store
uv run director run show <run_id>  # record, verdicts, and every tool call
uv run director resume <run_id>    # rebuild the agreed cut on its own timeline
```

`resume` builds onto `"<original timeline> (resume N)"` rather than replaying into the existing timeline, so it can never collide with or duplicate work you already have.

### Use `resolve-mcp` from any MCP client

The server stands on its own — point Claude Desktop, Claude Code, or any MCP client at it:

```bash
uv run resolve-mcp --backend davinci            # live Resolve, over stdio
uv run resolve-mcp --backend fake               # no Resolve needed
uv run resolve-mcp --backend davinci --allow-destructive   # enable gated ops
```

Example Claude Desktop entry:

```json
{
  "mcpServers": {
    "davinci-resolve": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/Davinci-MCP", "resolve-mcp", "--backend", "davinci"]
    }
  }
}
```

Server flags: `--backend {fake,davinci}`, `--allow-destructive` / `--no-allow-destructive`, `--transport stdio`, `--log-level {DEBUG,INFO,WARNING,ERROR}`. Anything you leave off falls back to `RESOLVE_MCP_*` environment variables (or `.env`), then to the defaults. Logs always go to stderr; stdout carries only the protocol.

**Conventions a client should know:**

- Times are **seconds from the start of the timeline**; the backend snaps them to whole frames and reports the snapped value back.
- Track indexes are **1-based across video tracks first, then audio** — on a fresh timeline `1` = V1 and `2` = A1. `get_timeline_state().tracks` lists the real mapping.
- Transform/crop values use **Resolve's Inspector units**: pan and anchor in pixels from centre, zoom as a multiplier (`1.0` = 100%), rotation in degrees, crop in pixels.
- Marker positions are relative to **the start of their item**.
- Every mutating tool returns `before`/`after`/`changed_paths`, plus `id_remap` when an operation had to recreate an item.

---

## Resolve bootstrap

Resolve's Python module lives inside the Resolve install. The server adds the standard location to `sys.path` automatically, so on a default install nothing is needed beyond enabling scripting. If Resolve lives somewhere custom, set these before launching the server:

| Variable | macOS | Windows | Linux |
| --- | --- | --- | --- |
| `RESOLVE_SCRIPT_API` | `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting` | `%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting` | `/opt/resolve/Developer/Scripting` |
| `RESOLVE_SCRIPT_LIB` | `/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so` | `C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll` | `/opt/resolve/libs/Fusion/fusionscript.so` |
| `PYTHONPATH` | `$RESOLVE_SCRIPT_API/Modules/` | `%RESOLVE_SCRIPT_API%\Modules\` | `$RESOLVE_SCRIPT_API/Modules/` |

Then: **Resolve → Preferences → General → External scripting using = Local**, and keep Resolve Studio running. The server connects lazily, so starting Resolve after the server is fine.

If Resolve isn't reachable, the server still boots and every tool call returns that checklist instead of a traceback — start Resolve and retry, no restart needed.

---

## Limitations

Honest about what the platform allows:

- **No fades, retimes or transitions on live Resolve.** The documented scripting API has no entry point for them, so `add_fade`, `set_speed` and `add_transition` exist only on the fake backend and are not advertised when you run `--backend davinci`. Apply them by hand in Resolve.
- **Moving a clip recreates it.** There is no reposition API, so `move_clip` (and ripple `insert_clip`) delete and re-append the item: its id changes — the delta's `id_remap` tells you the new one — and colour grades or Fusion comps on that item are not carried over.
- **Timeline item ids are per-session.** They come from Resolve's own `GetUniqueId()`, which is stable while the project is open, not across restarts.
- **`restart_app` is fake-only.** Resolve can quit itself but cannot relaunch itself.
- **The live path is documentation-verified, not yet hardware-verified.** Every live call is checked against Blackmagic's documented API by an offline harness modelled from that documentation, and the test suite fails if the backend calls anything outside it — but a manual smoke test on real Resolve Studio hardware is still pending. See `plan.md`.
- **Vision quality depends on the provider.** Gemini analyses the video file itself; OpenAI-compatible endpoints get sampled keyframes.

---

## Known issues

Open problems in *this* implementation, as opposed to the platform limits above. Nothing here is hidden by a passing test.

- **No hardware smoke test yet.** One detail of Blackmagic's reference is genuinely ambiguous and only real Resolve can settle it: whether `TimelineItem.AddMarker`'s frame is relative to the item or to the timeline. Reads and writes use the same convention, so a mismatch would show up as a marker in the wrong place rather than an error. Two other ambiguities used to sit here and now report themselves instead: `append_clip` reads the created item's length back (so an inclusive `clipInfo.endFrame` fails with both frame counts instead of quietly making every clip one frame long), and `GetRenderJobStatus` matches its keys case-insensitively and raises if there is no status field at all — defaulting would have reported a finished render as `queued` forever.
- **The two layers are not as isolated as the diagram suggests.** `director/mcp_client/client.py` imports `resolve_mcp` internals for its in-process stub path, and director's tests use `FakeResolveBackend` directly — yet `packages/director/pyproject.toml` does not declare `resolve-mcp` as a dependency. It only works because the uv workspace installs both. Installing `director` on its own would break the stub path.
- **Offline interactive mode understands a fixed vocabulary.** Without an LLM the REPL parses fade / opacity / speed / zoom / rotate / marker / move / delete instructions and says plainly when it doesn't understand something. It will not improvise, which is deliberate — but it does mean "make it feel dreamier" needs a provider configured.
- **stdio is the only transport.** `RESOLVE_MCP_TRANSPORT` accepts nothing else yet.
- **Render presets are a fixed set.** `mp4`, `mov`, `prores` and `dnxhr` map to format/codec hints that are matched against what your Resolve build actually offers; anything else needs a new entry in `_RENDER_FORMAT_HINTS`.
- **`insert_clip` refuses to split a clip.** Inserting in the middle of an existing item is rejected rather than silently overlapping it; cut the item first.
- **One test needs a working ffmpeg and skips without one.** `test_media_probe.py::test_reads_real_duration_and_rate` asks ffmpeg to write a real fixture clip into pytest's temp directory, so it skips on CI (no ffmpeg installed) and under WSL driving a Windows ffmpeg (which cannot write to the Linux temp path). The other 316 tests run everywhere.

---

## Project layout

```
Davinci-MCP/
├── packages/
│   ├── resolve-mcp/     # Layer 1 — MCP server (mcp, pydantic, structlog)
│   └── director/        # Layer 2 — orchestrator (google-genai, openai, librosa, typer, …)
├── .env.example
├── AGENTS.md            # working notes for AI agents in this repo
├── docs/
│   └── resolve-scripting-api.md   # the sourced API notes the live backend was written from
├── LICENSE              # MIT
├── DECISIONS.md         # the "why" behind the key design choices
├── plan.md              # remediation roadmap and its status
├── pyproject.toml       # uv workspace root
└── uv.lock
```

## Development

```bash
uv sync --all-extras --all-packages                          # install everything
uv run ruff check .                                          # lint
uv run mypy packages/resolve-mcp/src packages/director/src   # type-check (strict, scoped)
uv run pytest                                                # full suite, no Resolve or API key needed
```

The scoped mypy path is the canonical one — `mypy .` also walks the test tree, which is intentionally not strict-typed. CI runs exactly these three commands.

Tests run entirely against the fake backend plus a harness modelled on Blackmagic's documented scripting API, so CI never needs DaVinci Resolve or an API key. That documentation is checked in as [docs/resolve-scripting-api.md](docs/resolve-scripting-api.md) — every scripting method the live backend calls appears there with its sources, and `test_calls_only_documented_api` fails the build if the backend calls anything outside it.

---

## Built with

This project was built almost entirely by AI, and it's worth being clear about who did what:

- **🏗️ Execution — [MiniMax M3](https://www.minimax.io/), served via [NVIDIA NIM](https://www.nvidia.com/en-us/ai/).** MiniMax M3 was the primary building model — it wrote essentially the entire first version across both packages. Huge thanks to **NVIDIA NIM** for providing access to MiniMax and making the build possible.
- **🧭 Planning and remediation — Claude Opus.** The initial architecture was drafted with Opus, which later audited the build against Blackmagic's documentation and reworked the live backend, the planner and the review loop.

## License

[MIT](LICENSE) © 2026 Suryaansh.
