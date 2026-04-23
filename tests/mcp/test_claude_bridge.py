"""Tests for the Claude Code MCP bridge."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.mcp.claude_bridge import DEFAULT_SERVERS, _load_server_configs, load_mcp_tools
from openjarvis.tools._stubs import ToolSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(servers: dict) -> str:
    return json.dumps({"mcpServers": servers})


def _make_mock_client(tool_names: list[str]) -> MagicMock:
    client = MagicMock()
    client.list_tools.return_value = [
        ToolSpec(name=n, description=f"{n} tool", parameters={}) for n in tool_names
    ]
    return client


# ---------------------------------------------------------------------------
# _load_server_configs
# ---------------------------------------------------------------------------


class TestLoadServerConfigs:
    def test_returns_empty_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "openjarvis.mcp.claude_bridge._CONFIG_PATHS",
            [tmp_path / "missing.json"],
        )
        assert _load_server_configs() == {}

    def test_loads_single_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "settings.json"
        cfg_file.write_text(_make_config({"slack": {"command": "npx", "args": []}}))
        monkeypatch.setattr(
            "openjarvis.mcp.claude_bridge._CONFIG_PATHS", [cfg_file]
        )
        result = _load_server_configs()
        assert "slack" in result
        assert result["slack"]["command"] == "npx"

    def test_merges_multiple_files(self, tmp_path, monkeypatch):
        file_a = tmp_path / "a.json"
        file_b = tmp_path / "b.json"
        file_a.write_text(_make_config({"slack": {"command": "npx"}}))
        file_b.write_text(_make_config({"github": {"command": "npx"}}))
        monkeypatch.setattr(
            "openjarvis.mcp.claude_bridge._CONFIG_PATHS", [file_a, file_b]
        )
        result = _load_server_configs()
        assert "slack" in result
        assert "github" in result

    def test_later_file_overrides_earlier(self, tmp_path, monkeypatch):
        file_a = tmp_path / "a.json"
        file_b = tmp_path / "b.json"
        file_a.write_text(_make_config({"slack": {"command": "old"}}))
        file_b.write_text(_make_config({"slack": {"command": "new"}}))
        monkeypatch.setattr(
            "openjarvis.mcp.claude_bridge._CONFIG_PATHS", [file_a, file_b]
        )
        result = _load_server_configs()
        assert result["slack"]["command"] == "new"

    def test_skips_malformed_file(self, tmp_path, monkeypatch, caplog):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{")
        monkeypatch.setattr(
            "openjarvis.mcp.claude_bridge._CONFIG_PATHS", [bad_file]
        )
        result = _load_server_configs()
        assert result == {}


# ---------------------------------------------------------------------------
# load_mcp_tools
# ---------------------------------------------------------------------------


class TestLoadMcpTools:
    def _patch_configs(self, monkeypatch, configs: dict):
        monkeypatch.setattr(
            "openjarvis.mcp.claude_bridge._load_server_configs",
            lambda: configs,
        )

    def test_skips_server_not_in_config(self, monkeypatch):
        self._patch_configs(monkeypatch, {})
        tools, clients = load_mcp_tools(servers=["slack"])
        assert tools == []
        assert clients == []

    def test_skips_server_with_no_command(self, monkeypatch):
        self._patch_configs(monkeypatch, {"slack": {"command": None, "args": []}})
        tools, clients = load_mcp_tools(servers=["slack"])
        assert tools == []
        assert clients == []

    def test_connects_and_returns_tools(self, monkeypatch):
        self._patch_configs(
            monkeypatch,
            {"slack": {"command": "npx", "args": ["-y", "slack-mcp-server"], "env": {}}},
        )
        mock_client = _make_mock_client(["slack_post", "slack_search"])

        with (
            patch("openjarvis.mcp.claude_bridge.StdioTransport"),
            patch(
                "openjarvis.mcp.claude_bridge.MCPClient", return_value=mock_client
            ),
        ):
            tools, clients = load_mcp_tools(servers=["slack"])

        assert len(tools) == 2
        tool_names = {t.spec.name for t in tools}
        assert tool_names == {"slack_post", "slack_search"}
        assert clients == [mock_client]

    def test_returns_client_for_cleanup(self, monkeypatch):
        self._patch_configs(
            monkeypatch,
            {"github": {"command": "npx", "args": [], "env": {}}},
        )
        mock_client = _make_mock_client(["search_repos"])

        with (
            patch("openjarvis.mcp.claude_bridge.StdioTransport"),
            patch(
                "openjarvis.mcp.claude_bridge.MCPClient", return_value=mock_client
            ),
        ):
            _, clients = load_mcp_tools(servers=["github"])

        assert mock_client in clients

    def test_failed_server_does_not_block_others(self, monkeypatch):
        self._patch_configs(
            monkeypatch,
            {
                "bad": {"command": "does-not-exist", "args": [], "env": {}},
                "good": {"command": "npx", "args": [], "env": {}},
            },
        )
        mock_client = _make_mock_client(["good_tool"])

        def selective_transport(cmd, env=None):
            if cmd[0] == "does-not-exist":
                raise FileNotFoundError("not found")
            return MagicMock()

        with (
            patch("openjarvis.mcp.claude_bridge.StdioTransport", side_effect=selective_transport),
            patch(
                "openjarvis.mcp.claude_bridge.MCPClient", return_value=mock_client
            ),
        ):
            tools, clients = load_mcp_tools(servers=["bad", "good"])

        assert len(tools) == 1
        assert tools[0].spec.name == "good_tool"

    def test_env_overrides_merged_with_os_env(self, monkeypatch):
        """Server-specific env vars are passed through to StdioTransport."""
        self._patch_configs(
            monkeypatch,
            {
                "slack": {
                    "command": "npx",
                    "args": [],
                    "env": {"SLACK_TOKEN": "xoxp-test"},
                }
            },
        )
        mock_client = _make_mock_client([])
        captured = {}

        def capture_transport(cmd, env=None):
            captured["env"] = env
            return MagicMock()

        with (
            patch(
                "openjarvis.mcp.claude_bridge.StdioTransport",
                side_effect=capture_transport,
            ),
            patch(
                "openjarvis.mcp.claude_bridge.MCPClient", return_value=mock_client
            ),
        ):
            load_mcp_tools(servers=["slack"])

        assert captured["env"].get("SLACK_TOKEN") == "xoxp-test"

    def test_uses_default_servers_when_none_specified(self, monkeypatch):
        self._patch_configs(monkeypatch, {})
        # Should not raise; just returns empty because configs are empty
        tools, clients = load_mcp_tools()
        assert tools == []

    def test_default_servers_list(self):
        assert "slack" in DEFAULT_SERVERS
        assert "github" in DEFAULT_SERVERS
        assert "apple-mcp" in DEFAULT_SERVERS
