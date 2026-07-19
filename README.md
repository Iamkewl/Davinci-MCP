# davinci-mcp

Two-layer system that automates video editing in **DaVinci Resolve Studio**.

| Layer | Package | Role | Talks to whom |
|-------|---------|------|---------------|
| 1 | [`resolve-mcp`](packages/resolve-mcp/) | **MCP server**: drives DaVinci Resolve's scripting API. | Listens for MCP calls from `director`. |
| 2 | [`director`](packages/director/) | **MCP client** + agent orchestrator: Contextualizer → Planner ↔ Director → Editor. Uses **Google Gemini** for vision/reasoning and **librosa** for audio. | Calls `resolve-mcp` over stdio (MCP). |

The two layers communicate **only** via the Model Context Protocol. No shared process,
no shared globals, no scraped Python files cross-imported across the boundary.

* Status: v0.1.0
* Python: 3.11+
* Workspace manager: [`uv`](https://docs.astral.sh/uv/)

---

## Why this exists (and what it replaces)

The previous build:

- Vendored a third-party Resolve MCP server as a raw ZIP.
- Exposed ~27 action-dispatch tools hiding 342 ops behind one entry point — LLMs
  hallucinated arguments.
- Did not read state back after mutations, so failed edits went unnoticed.
- Silently force-accepted a low-quality plan on max planner iterations.
- Had no persistence/observability; runs could not be inspected or resumed.
- Destructive ops had no confirmation gate.

This rewrite fixes all of those:

- **First-party** MCP server (`resolve-mcp`). No vendored zip.
- **One tool per operation**, type-hinted via FastMCP. No `timeline(action="append_clip", …)` dispatch.
- **State-delta returns** on every mutation; the editor reads back state to verify
  each edit landed.
- **Honest Director verdicts** — `APPROVED` / `ACCEPTED_WITH_WARNINGS` / `FAILED`.
  Never silently force-accepted; on max iterations we surface the latest verdict
  with `status=failed`.
- **SQLite + JSONL run store**; per-run events replayable. `--resume <run_id>` lands
  at the last checkpoint.
- **Destructive gate**: configuration flag `--allow-destructive` + per-call
  `confirm=true` argument. Both must be set.

See [`DECISIONS.md`](DECISIONS.md) for the rationale on each non-obvious choice.

---

## Resolve bootstrap (required before `resolve-mcp` can drive the real app)

DaVinci Resolve ships a Python API that lives outside any normal site-packages.
You **must** add its folder to `PYTHONPATH` and launch Resolve in *Local scripting*
mode. Without these, `import DaVinciResolveScript` fails inside the server.

### One-time: Resolve preferences
Resolve → **Preferences** → **General** → **External scripting using** = **Local**.

### Per-shell env vars

| Variable | Windows (PowerShell) | macOS (zsh/bash) |
|----------|------------------------|------------------|
| `RESOLVE_SCRIPT_API` | `C:\Program Files\Blackmagic Design\DaVinci Resolve\Developer\Scripting\API` | `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/API` |
| `RESOLVE_SCRIPT_LIB` | `…\DaVinci Resolve\Developer\Scripting\Lib\resolve_lite.exe` *(Studio: no `_lite`)* | `…/DaVinci Resolve/Developer/Scripting/Lib/resolve_lite.app` |
| `PYTHONPATH` | append `%RESOLVE_SCRIPT_API%` | export `PYTHONPATH="$RESOLVE_SCRIPT_API:$PYTHONPATH"` |

See [`.env.example`](.env.example) for an annotated template. **Never commit `.env`.**

The server's `import DaVinciResolveScript` is guarded, so it boots fine even
without Resolve — only tools that *call* the backend raise a clear
`resolve_unavailable` error.

---

## Development

```bash
# install the workspace (creates .venv at the repo root)
uv sync

# run all tests
uv run pytest

# lint / typecheck
uv run ruff check .
uv run mypy packages/resolve-mcp/src packages/director/src
```

### Running the server alone

```bash
# stdio, in-memory fake (no Resolve needed for dev / CI)
uv run resolve-mcp --backend fake

# stdio against the real Resolve (requires the bootstrap above)
uv run resolve-mcp --backend davinci

# enable destructive tools (quit_app, restart_app, delete_timeline, delete_media)
uv run resolve-mcp --allow-destructive
```

### Running the orchestrator

```bash
# auto: clips dir + music + prompt -> beat-synced timeline
uv run director auto ./clips --music ./music.mp3 --prompt "high-energy 30s reel"

# offline (`--fast`): run planner + director deterministically (no Gemini key required)
uv run director auto ./clips --fast

# interactive REPL — type natural-language edits; the director produces and applies them
uv run director interactive --backend fake

# inspect a past run
uv run director run list
uv run director run show <run_id>
```

---

## Architecture in one diagram

```
              ┌─────────────────────────────────────┐
              │  director  (pure MCP CLIENT)        │
              │                                     │
   Gemini ──► │  Contextualizer ─► Planner◄─►Director│
              │                       │              │
              │                       ▼              │
              │                     Editor           │
              │                       │              │
              └───────────────────────┼──────────────┘
                                      │  MCP / stdio
                                      ▼
              ┌─────────────────────────────────────┐
              │  resolve-mcp  (MCP SERVER)          │
              │                                     │
              │  tools ──► ResolveBackend (protocol)│
              │              ├─ DaVinciResolveBackend│
              │              └─ FakeResolveBackend   │
              └─────────────────────────────────────┘
```

---

## Tool surface (the 28 tools `resolve-mcp` exposes)

* **Project** — `create_project`, `open_project`, `save_project`, `get_project_info`
* **Media** — `import_media`, `list_media_pool`, `create_bin`
* **Timeline** — `create_timeline`, `get_timeline_state`, `append_clip`,
  `insert_clip`, `delete_clip`, `move_clip`
* **Item** — `set_transform`, `set_crop`, `set_composite_mode`, `set_opacity`,
  `add_fade`, `set_speed`, `add_marker`
* **Effects** — `add_transition`
* **Render** — `add_render_job`, `start_render`, `get_render_status`
* **Destructive (gated)** — `quit_app`, `restart_app`, `delete_timeline`, `delete_media`

Resources:
* `resolve://project`
* `resolve://media-pool`
* `resolve://timeline/current`

---

## Manual live-Resolve smoke test

This is intentionally **not** part of the CI suite — it needs a real Resolve install.
Documented for the human running it on their workstation:

1. Complete the [bootstrap above](#resolve-bootstrap-required-before-resolve-mcp-can-drive-the-real-app).
2. From the workspace root, with `.env` pointing at a real Resolve:

   ```bash
   uv run resolve-mcp --backend davinci
   ```
3. In a separate terminal, validate the protocol round-trips:

   ```bash
   uv run python - <<'PY'
   import asyncio, json
   from director.mcp_client import StdioResolveClient

   async def main():
       c = StdioResolveClient.default(backend="davinci", uv_project=".")
       await c.start()
       print(await c.list_tools())
       print(json.dumps(await c.call_tool(
           "create_project", {"name": "smoke", "fps": 24.0}), indent=2))
       await c.close()
   asyncio.run(main())
   PY
   ```
4. Run the auto pipeline end-to-end against live Resolve:

   ```bash
   uv run director auto ./clips --uv-project . --backend davinci \
       --prompt "test the wiring"
   ```

   (Without `--backend davinci` the auto path default-installed `--backend fake` skips
   the resolve-mcp subprocess and runs in-process — useful for rehearsal.)
5. In the Resolve UI, verify the project `smoke` + at least one import + clip landed.

The CI covers the same graph **except** `import DaVinciResolveScript` via the
`tests/fake_resolve.py` record/replay harness in `tests/test_live_backend.py`. CI proves:

* argument shapes match the scripting API,
* method names exist,
* everything wires up.

A real Resolve is only required to verify the *exact* scripting-API behavior we just
broader-strokes from the docs.

---

## Inspecting / resuming a run

Each `director` run writes:

- `runs.sqlite` — runs, plans, verdicts, tool calls.
- `events.jsonl` — append-only event log.

```bash
uv run director run show <run_id>     # full record
uv run director run list              # recent 50
```

Resumability surface primitives (`store.get_run`, `store.list_verdicts`,
`store.list_tool_calls`) live in `director.store.RunStore`. A `--resume` flag
lands on `director` in Phase 5+ polish.

---

## License

See repository.
