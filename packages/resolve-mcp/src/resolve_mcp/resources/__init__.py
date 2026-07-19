"""MCP resources: read-only views of state.

Resources are URI-addressable snapshots a client can subscribe to. We expose the three
top-level slices an orchestrator or a human operator cares about most — project,
media pool, current timeline — and we keep emitting them as plain JSON for cheap
diffing on the client side.
"""

from __future__ import annotations

import json
from typing import Any

from ..backend import ResolveBackend


def project_resource(backend: ResolveBackend) -> str:
    """resolve://project — full project state."""
    return json.dumps(backend.current_project().model_dump(mode="json"), indent=2)


def media_pool_resource(backend: ResolveBackend) -> str:
    """resolve://media-pool — bins and clips."""
    return json.dumps(backend.list_media_pool().model_dump(mode="json"), indent=2)


def timeline_resource(backend: ResolveBackend) -> str:
    """resolve://timeline/current — full timeline state incl. items/transforms."""
    return json.dumps(backend.get_timeline_state().model_dump(mode="json"), indent=2)


__all__ = ["media_pool_resource", "project_resource", "timeline_resource"]


def all_resources(backend: ResolveBackend) -> dict[str, Any]:
    """Return all three resources keyed by URI — useful for tests and a UI dump."""
    return {
        "resolve://project": project_resource(backend),
        "resolve://media-pool": media_pool_resource(backend),
        "resolve://timeline/current": timeline_resource(backend),
    }
