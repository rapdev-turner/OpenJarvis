"""
Jarvis voice loop — press Enter to speak, Enter again to stop.

Uses an in-process OrchestratorAgent backed by Claude (default) or GPT-4o
(set OPENAI_API_KEY and pass --model gpt-4o). Conversation history, personal
context, Obsidian vault retrieval, and MCP tools (Slack, GitHub, Apple) all
persist across every turn within a session.

Usage:
    uv run python jarvis_voice.py                  # Claude sonnet (default)
    uv run python jarvis_voice.py --model gpt-4o   # GPT-4o (needs OPENAI_API_KEY)
    uv run python jarvis_voice.py --no-mcp          # skip MCP server startup
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

logging.basicConfig(level=logging.WARNING)

SAMPLE_RATE = 16_000
CHANNELS = 1
WHISPER_MODEL = "base"
SOUL_PATH = Path.home() / ".openjarvis" / "SOUL.md"
USER_PATH = Path.home() / ".openjarvis" / "USER.md"

# Lazy-loaded singletons — initialized on first use
_whisper_model = None
_tts_pipeline = None


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


def _resolve_model(requested: Optional[str]) -> Tuple[str, str]:
    """Return (engine_key, model_id) for the requested model name.

    Supports:
      claude-*         → cloud engine, Anthropic model
      gpt-* / o3-*     → litellm engine, openai/model
      default (None)   → claude-sonnet-4-6 via cloud
    """
    if requested is None:
        return "cloud", "claude-sonnet-4-6"

    lower = requested.lower()
    if lower.startswith(("gpt-", "o3-", "o1-", "openai/")):
        model = lower if lower.startswith("openai/") else f"openai/{lower}"
        if not os.environ.get("OPENAI_API_KEY"):
            print("[Warning] OPENAI_API_KEY not set — falling back to Claude")
            return "cloud", "claude-sonnet-4-6"
        return "litellm", model

    if lower.startswith("anthropic/"):
        return "litellm", lower
    return "cloud", lower


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def record_until_enter() -> np.ndarray:
    print("\n[Listening... press Enter to stop]")
    chunks: List[np.ndarray] = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    ):
        input()

    return (
        np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype="float32")
    )


def transcribe(audio: np.ndarray) -> str:
    global _whisper_model
    from faster_whisper import WhisperModel

    if _whisper_model is None:
        print("[Loading Whisper...]")
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device="auto", compute_type="float32"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        tmp_path = f.name

    segments, _ = _whisper_model.transcribe(tmp_path)
    return "".join(seg.text for seg in segments).strip()


def speak(text: str) -> None:
    global _tts_pipeline
    from kokoro import KPipeline

    if _tts_pipeline is None:
        _tts_pipeline = KPipeline(lang_code="b")

    samples = []
    for _, _, audio in _tts_pipeline(text, voice="bm_george"):
        samples.append(audio)

    if not samples:
        return

    combined = np.concatenate(samples)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, combined, 24_000)
        tmp_path = f.name

    subprocess.run(["afplay", tmp_path], check=False)


# ---------------------------------------------------------------------------
# Agent + tool setup
# ---------------------------------------------------------------------------


def _load_system_prompt() -> str:
    parts = []
    if SOUL_PATH.exists():
        parts.append(SOUL_PATH.read_text().strip())
    profile = USER_PATH.read_text().strip() if USER_PATH.exists() else ""
    if profile and profile != "# User Profile":
        parts.append(f"## User Profile\n\n{profile}")
    return "\n\n---\n\n".join(parts)


def _get_memory_backend(config):
    try:
        import openjarvis.tools.storage  # noqa: F401
        from openjarvis.core.registry import MemoryRegistry

        key = config.memory.default_backend
        if not MemoryRegistry.contains(key):
            return None
        backend = (
            MemoryRegistry.create(key, db_path=config.memory.db_path)
            if key == "sqlite"
            else MemoryRegistry.create(key)
        )
        if hasattr(backend, "count") and backend.count() == 0:
            if hasattr(backend, "close"):
                backend.close()
            return None
        return backend
    except Exception:
        return None


def build_agent(engine_key: str, model: str, enable_mcp: bool = True):
    """Build the OrchestratorAgent.

    Returns (agent, memory_backend, mcp_clients, config).
    mcp_clients must be closed on shutdown.
    """
    import openjarvis.agents  # noqa: F401
    import openjarvis.tools  # noqa: F401
    import openjarvis.tools.storage  # noqa: F401
    from openjarvis.core.config import load_config
    from openjarvis.core.events import EventBus
    from openjarvis.core.registry import AgentRegistry, ToolRegistry
    from openjarvis.engine import get_engine
    from openjarvis.intelligence import register_builtin_models

    config = load_config()
    register_builtin_models()

    resolved = get_engine(config, engine_key)
    if resolved is None:
        raise RuntimeError(
            f"Engine '{engine_key}' not available. "
            "Check ANTHROPIC_API_KEY / OPENAI_API_KEY."
        )
    _, engine = resolved

    # Native OpenJarvis tools
    native_tool_names = [
        "user_profile_manage",
        "web_search",
        "shell_exec",
        "file_read",
        "calculator",
    ]
    tools = []
    for name in native_tool_names:
        if ToolRegistry.contains(name):
            tools.append(ToolRegistry.get(name)())

    # Retrieval tool (knowledge base / Obsidian vault)
    memory_backend = _get_memory_backend(config)
    if memory_backend is not None and ToolRegistry.contains("retrieval"):
        tools.append(ToolRegistry.get("retrieval")(backend=memory_backend))
        print(f"[KB] {memory_backend.count()} chunks indexed")

    # MCP tools (Slack, GitHub, Apple)
    mcp_clients = []
    if enable_mcp:
        try:
            from openjarvis.mcp.claude_bridge import load_mcp_tools

            mcp_tools, mcp_clients = load_mcp_tools()
            tools.extend(mcp_tools)
        except Exception as exc:
            print(f"[MCP] Bridge failed: {exc}")

    system_prompt = _load_system_prompt()

    agent_cls = AgentRegistry.get("orchestrator")
    agent = agent_cls(
        engine,
        model,
        bus=EventBus(),
        tools=tools,
        max_turns=20,
        temperature=0.7,
        max_tokens=4096,
        interactive=False,
        confirm_callback=lambda _: True,
        system_prompt=system_prompt,
    )

    return agent, memory_backend, mcp_clients, config


# ---------------------------------------------------------------------------
# Per-turn memory injection
# ---------------------------------------------------------------------------


def _inject_memory(query: str, memory_backend, config) -> Optional[str]:
    if memory_backend is None:
        return None
    try:
        from openjarvis.tools.storage.context import ContextConfig, format_context

        ctx_cfg = ContextConfig(
            top_k=config.memory.context_top_k,
            min_score=config.memory.context_min_score,
            max_context_tokens=config.memory.context_max_tokens,
        )
        results = memory_backend.search(query, top_k=ctx_cfg.top_k)
        hits = [r for r in results if r.score >= ctx_cfg.min_score]
        return format_context(hits) if hits else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session (persistent conversation across turns)
# ---------------------------------------------------------------------------


class Session:
    def __init__(self):
        self._history: List[Tuple[str, str]] = []  # (role, content)

    def ask(self, query: str, agent, memory_backend, config) -> str:
        from openjarvis.agents._stubs import AgentContext
        from openjarvis.core.types import Message, Role

        ctx = AgentContext()

        # Replay conversation history into context
        for role_str, content in self._history:
            role = Role.USER if role_str == "user" else Role.ASSISTANT
            ctx.conversation.add(Message(role=role, content=content))

        # Inject semantically relevant vault / profile context
        kb_context = _inject_memory(query, memory_backend, config)
        if kb_context:
            ctx.conversation.add(
                Message(
                    role=Role.SYSTEM,
                    content="Relevant context from the knowledge base:\n\n"
                    + kb_context,
                )
            )

        result = agent.run(query, context=ctx)

        self._history.append(("user", query))
        self._history.append(("assistant", result.content))

        return result.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Jarvis voice assistant")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use: claude-sonnet-4-6 (default), gpt-4o, claude-opus-4-6, etc.",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Skip MCP server startup (faster, no Slack/GitHub/Apple tools)",
    )
    args = parser.parse_args()

    engine_key, model = _resolve_model(args.model)

    print("=== Jarvis Voice Mode ===")
    print(f"Model: {model}  |  Engine: {engine_key}")
    print("Initializing...")

    agent, memory_backend, mcp_clients, config = build_agent(
        engine_key,
        model,
        enable_mcp=not args.no_mcp,
    )
    session = Session()

    print("Ready. Press Enter to start speaking. Type 'quit' to exit.\n")
    speak("Good day. Jarvis online. How may I assist you?")

    try:
        while True:
            try:
                cmd = input("\nPress Enter to speak (or type 'quit'): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                break

            if cmd == "quit":
                speak("Goodbye.")
                break

            audio = record_until_enter()
            if audio.shape[0] < SAMPLE_RATE * 0.3:
                print("[Too short, try again]")
                continue

            print("[Transcribing...]")
            query = transcribe(audio)
            if not query:
                print("[Couldn't understand, try again]")
                continue

            print(f"You: {query}")
            print("[Thinking...]")

            try:
                response = session.ask(query, agent, memory_backend, config)
            except Exception as exc:
                response = f"I encountered an error: {exc}"
                print(f"[Error: {exc}]")

            print(f"Jarvis: {response}\n")
            speak(response)

    finally:
        for client in mcp_clients:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
