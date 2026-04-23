"""Load MCP servers from Claude Code config as OpenJarvis tools.

Reads ~/.claude/settings.json and ~/.claude/mcp-configs/mcp-servers.json,
spawns each requested MCP server as a subprocess via StdioTransport, and
returns their tools as MCPToolAdapter objects ready for use in any agent.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openjarvis.mcp.client import MCPClient
from openjarvis.mcp.transport import StdioTransport
from openjarvis.tools._stubs import BaseTool
from openjarvis.tools.mcp_adapter import MCPToolProvider

logger = logging.getLogger(__name__)

# Servers that work headlessly (no browser OAuth needed).
# google-workspace requires a browser auth flow so is excluded by default.
DEFAULT_SERVERS = ["slack", "github", "apple-mcp"]

_CONFIG_PATHS = [
    Path.home() / ".claude" / "settings.json",
    Path.home() / ".claude" / "mcp-configs" / "mcp-servers.json",
]


def _load_server_configs() -> Dict[str, dict]:
    """Merge MCP server definitions from all Claude config files."""
    merged: Dict[str, dict] = {}
    for path in _CONFIG_PATHS:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            merged.update(data.get("mcpServers", {}))
        except Exception as exc:
            logger.debug("Failed to load MCP config from %s: %s", path, exc)
    return merged


def load_mcp_tools(
    servers: Optional[List[str]] = None,
) -> Tuple[List[BaseTool], List[MCPClient]]:
    """Connect to Claude Code MCP servers and return their tools.

    Parameters
    ----------
    servers:
        Server names to connect. Defaults to DEFAULT_SERVERS.

    Returns
    -------
    tools:
        List of BaseTool-compatible MCPToolAdapter objects.
    clients:
        Open MCPClient instances — caller must keep these alive as long as
        the tools are in use, and call client.close() on shutdown.
    """
    target = servers or DEFAULT_SERVERS
    configs = _load_server_configs()

    tools: List[BaseTool] = []
    clients: List[MCPClient] = []

    for name in target:
        cfg = configs.get(name)
        if not cfg:
            logger.warning("MCP server '%s' not found in Claude config — skipping", name)
            continue

        command = cfg.get("command")
        args = cfg.get("args", [])
        env_overrides = cfg.get("env", {})

        if not command:
            logger.warning("MCP server '%s' has no command — skipping", name)
            continue

        # Merge server-specific env vars on top of current environment
        full_env = {**os.environ, **env_overrides}

        try:
            transport = StdioTransport([command] + args, env=full_env)
            client = MCPClient(transport)
            client.initialize()

            server_tools = MCPToolProvider(client).discover()
            clients.append(client)
            tools.extend(server_tools)
            print(f"[MCP] {name}: {len(server_tools)} tools")
        except Exception as exc:
            logger.warning("Failed to connect to MCP server '%s': %s", name, exc)
            print(f"[MCP] {name}: failed — {exc}")

    return tools, clients


__all__ = ["load_mcp_tools", "DEFAULT_SERVERS"]
