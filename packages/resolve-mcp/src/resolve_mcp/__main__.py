"""Allow `python -m resolve_mcp` (used by director when uv is not on PATH)."""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
