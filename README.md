# DaVinci-MCP

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
  │  • Gemini vision (clips)    │        │  • 28 typed tools            │
  │  • librosa beat detection   │        │  • drives Resolve's API      │
  │  • plan → review → execute  │        │  • state-delta verification  │
  │  • SQLite + JSONL run store │        │  • fake backend for testing  │
  └─────────────────────────────┘        └──────────────────────────────┘
```

### Layer 1 — `resolve-mcp` (the hands)

A first-party MCP **server** that exposes DaVinci Resolve's scripting API as **28 individual, type-hinted tools** — one function per operation (`append_clip`, `set_transform`, `add_transition`, `add_render_job`, …) instead of a handful of overloaded string-dispatch tools. Every state-changing tool returns a snapshot of what changed, so the caller can *verify* an edit actually landed rather than hoping it did. Destructive operations (`quit_app`, `restart_app`, `delete_timeline`, `delete_media`) are gated behind an explicit `--allow-destructive` flag *and* a per-call confirmation.

It ships with a **`FakeResolveBackend`** that models project/timeline state in memory — so you (and CI) can run and test the whole thing without DaVinci Resolve installed.

### Layer 2 — `director` (the brain)

An MCP **client** and agent orchestrator that runs the creative pipeline:

1. **Contextualize** — Google Gemini analyzes each clip to understand what's in it.
2. **Listen** — `librosa` detects tempo, beats, and onsets in your music.
3. **Plan → Review** — a planner drafts a beat-synced timeline; a director critiques it and returns an honest verdict: `APPROVED`, `ACCEPTED_WITH_WARNINGS`, or `FAILED` (no silent rubber-stamping).
4. **Execute** — the approved plan is built in Resolve via `resolve-mcp`, one verified edit at a time.

Every run is recorded to a **SQLite + JSONL run store** so you can inspect exactly what happened — and resume it later.

---

## Features

- 🎬 **Clips + music → finished timeline**, automatically.
- 🧩 **28 typed MCP tools** covering projects, media, timelines, effects, and rendering.
- ✅ **Verified edits** — mutations return state deltas; no silent failures.
- 🔎 **Honest reviews** — the director says when a cut isn't good enough.
- 💾 **Inspectable, resumable runs** — full history in SQLite + JSONL.
- 🛡️ **Safe by default** — destructive ops are double-gated.
- 🧪 **Runs without Resolve** — fake backend + a deterministic offline mode for testing.
- 💬 **Interactive mode** — refine the cut conversationally, not just one-shot.

---

## Requirements

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (workspace & dependency manager)
- **DaVinci Resolve Studio 18.5+** — *for real editing.* Studio only; the free edition has no external scripting. Enable it under **Preferences → General → External scripting using = Local**.
- **A Gemini API key** — for the full pipeline (vision + planning). Not needed in `--fast`/offline mode.
- **FFmpeg** on your PATH is recommended for broad audio format support.

> You can try everything below **without** Resolve or a Gemini key using the `fake` backend and `--fast` mode.

---

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/Iamkewl/Davinci-MCP.git
cd Davinci-MCP

# 2. Install the workspace
uv sync

# 3. (Optional) add your Gemini key for the full pipeline
cp .env.example .env
#   then set GEMINI_API_KEY=... in .env

# 4. Try it end-to-end with NO Resolve and NO API key:
uv run director auto ./clips --music ./music.mp3 \
    --prompt "high-energy 30s reel" --fast
```

`--fast` uses a deterministic planner/director (no Gemini) and the default `fake` backend (no Resolve), so it's the ideal way to see the flow before wiring up the real app.

---

## Usage

### Auto mode — clips + music → timeline

```bash
# Offline dry run (fake backend, deterministic planning)
uv run director auto ./clips -m ./music.mp3 -p "moody cinematic edit" --fast

# The real thing: drive a running DaVinci Resolve Studio
uv run director auto ./clips -m ./music.mp3 -p "moody cinematic edit" \
    --backend davinci --uv-project packages/resolve-mcp
```

| Option | Meaning |
| --- | --- |
| `clips_dir` | Directory of source clips (positional) |
| `--music`, `-m` | Music track to sync to |
| `--prompt`, `-p` | Your brief (default: `high-energy 30s reel`) |
| `--backend` | `fake` (default) or `davinci` |
| `--uv-project` | Path to `resolve-mcp` so director can launch the server |
| `--fast` | Skip Gemini; deterministic planner/director |

### Interactive mode — refine conversationally

```bash
uv run director interactive --fast
# ...or against real Resolve:
uv run director interactive --backend davinci --uv-project packages/resolve-mcp
```

### Inspect your runs

```bash
uv run director run list           # every run in the store
uv run director run show <run_id>  # record, verdicts, and every tool call
```

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
      "args": ["run", "resolve-mcp", "--backend", "davinci"],
      "cwd": "/path/to/Davinci-MCP/packages/resolve-mcp"
    }
  }
}
```

Server flags: `--backend {fake,davinci}`, `--allow-destructive`, `--transport stdio`, `--log-level {DEBUG,INFO,WARNING,ERROR}`.

---

## Project layout

```
Davinci-MCP/
├── packages/
│   ├── resolve-mcp/     # Layer 1 — MCP server (mcp, pydantic, structlog)
│   └── director/        # Layer 2 — orchestrator (google-genai, librosa, typer, …)
├── .env.example
├── DECISIONS.md         # the "why" behind the key design choices
├── pyproject.toml       # uv workspace root
└── uv.lock
```

## Development

```bash
uv sync                 # install everything
uv run pytest           # run the test suites (no Resolve required)
uv run ruff check .     # lint
uv run mypy .           # type-check (strict)
```

Tests run entirely against the fake backend and a record/replay harness, so CI never needs DaVinci Resolve or a Gemini key.

---

## Built with

This project was built almost entirely by AI, and it's worth being clear about who did what:

- **🏗️ Execution — [MiniMax M3](https://www.minimax.io/), served via [NVIDIA NIM](https://www.nvidia.com/en-us/ai/).** MiniMax M3 was the primary building model — it wrote essentially the entire codebase across both packages. Huge thanks to **NVIDIA NIM** for providing access to MiniMax and making the build possible.
- **🧭 Planning — Claude Opus.** The initial architecture and project plan were drafted with Opus before a line of code was written.

---

## License

Open source. See [`LICENSE`](LICENSE) in the repository.
