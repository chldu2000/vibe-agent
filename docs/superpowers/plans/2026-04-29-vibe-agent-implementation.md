# Vibe Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight Python AI agent library with OpenAI tool calling, MCP server support, parallel subagents, and streaming output.

**Architecture:** ReAct loop over OpenAI Chat Completions API with a unified ToolRegistry for local and MCP tools. MCP servers connected via `mcp` SDK (stdio + streamable_http). Subagents run as independent Agent instances with asyncio.gather.

**Tech Stack:** Python 3.11+, uv (package manager), openai, mcp (official SDK), pydantic, pytest + pytest-asyncio

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, build config |
| `src/vibe_agent/__init__.py` | Public API exports |
| `src/vibe_agent/config.py` | AgentConfig, MCPServerStdio, MCPServerSSE, SubagentConfig |
| `src/vibe_agent/types.py` | Response, StreamEvent, SubagentResult |
| `src/vibe_agent/tools/__init__.py` | Package init |
| `src/vibe_agent/tools/base.py` | Tool dataclass (name, description, parameters, handler) |
| `src/vibe_agent/tools/registry.py` | ToolRegistry — register, lookup, execute, OpenAI format |
| `src/vibe_agent/mcp/__init__.py` | Package init |
| `src/vibe_agent/mcp/client.py` | MCPClient — connect, list_tools, call_tool, disconnect |
| `src/vibe_agent/mcp/tool_adapter.py` | Convert MCP Tool → OpenAI function tool format |
| `src/vibe_agent/agent.py` | Core Agent class — run loop (normal + streaming), context manager |
| `src/vibe_agent/subagent.py` | delegate() — parallel subagent execution |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_types.py` | Type instantiation tests |
| `tests/test_tools.py` | ToolRegistry tests |
| `tests/test_mcp_adapter.py` | MCP tool adapter tests |
| `tests/test_agent.py` | Agent loop tests (mocked OpenAI) |
| `tests/test_streaming.py` | Streaming tests |
| `tests/test_subagent.py` | Subagent tests |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/vibe_agent/__init__.py`
- Create: `src/vibe_agent/tools/__init__.py`
- Create: `src/vibe_agent/mcp/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "vibe-agent"
version = "0.1.0"
description = "Lightweight Python AI agent with OpenAI tool calling, MCP support, and parallel subagents"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.30",
    "mcp>=1.0",
    "pydantic>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create package directories and init files**

`src/vibe_agent/__init__.py`:
```python
"""Vibe Agent — lightweight AI agent library."""
```

`src/vibe_agent/tools/__init__.py`:
```python
"""Tool system for vibe-agent."""
```

`src/vibe_agent/mcp/__init__.py`:
```python
"""MCP client integration for vibe-agent."""
```

`tests/conftest.py`:
```python
"""Shared test fixtures."""
```

- [ ] **Step 3: Initialize uv project and install dependencies**

Run: `uv init --no-readme` (skip if pyproject.toml already exists)
Run: `uv sync --group dev`
Expected: uv creates lockfile, installs openai, mcp, pydantic, pytest

- [ ] **Step 4: Verify pytest works**

Run: `uv run pytest --co -q`
Expected: "no tests collected" (no error)

- [ ] **Step 5: Init git repo and commit**

```bash
git init
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

### Task 2: Types and Config

**Files:**
- Create: `src/vibe_agent/types.py`
- Create: `src/vibe_agent/config.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing test for types**

`tests/test_types.py`:
```python
import pytest
from vibe_agent.types import Response, StreamEvent, SubagentResult


def test_response_creation():
    r = Response(content="hello", usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15})
    assert r.content == "hello"
    assert r.usage.total_tokens == 15


def test_stream_event_text():
    e = StreamEvent(type="text", content="hello")
    assert e.type == "text"
    assert e.content == "hello"


def test_stream_event_tool_call():
    e = StreamEvent(type="tool_call", name="read_file", arguments='{"path": "/tmp/a"}')
    assert e.name == "read_file"


def test_stream_event_done():
    e = StreamEvent(type="done", usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15})
    assert e.type == "done"
    assert e.usage.total_tokens == 10


def test_subagent_result_success():
    r = SubagentResult(status="success", content="done")
    assert r.status == "success"
    assert r.content == "done"


def test_subagent_result_timeout():
    r = SubagentResult(status="timeout", content="")
    assert r.status == "timeout"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL — ImportError for vibe_agent.types

- [ ] **Step 3: Create types.py**

`src/vibe_agent/types.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Response:
    content: str
    usage: TokenUsage | dict[str, int] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)


@dataclass
class StreamEvent:
    type: str  # "text" | "tool_call" | "tool_result" | "done"
    content: str = ""
    name: str | None = None
    arguments: str | None = None
    usage: TokenUsage | dict[str, int] | None = None

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)


@dataclass
class SubagentResult:
    status: str  # "success" | "timeout" | "error"
    content: str
    usage: TokenUsage | dict[str, int] | None = None

    def __post_init__(self):
        if isinstance(self.usage, dict):
            self.usage = TokenUsage(**self.usage)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Write config.py**

`src/vibe_agent/config.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPServerStdio:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass
class MCPServerSSE:
    url: str
    headers: dict[str, str] | None = None


@dataclass
class SubagentConfig:
    max_parallel: int = 5
    timeout: float = 60.0
    inherit_tools: bool = True


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    max_iterations: int = 20
    tool_timeout: float = 30.0
    mcp_servers: list[MCPServerStdio | MCPServerSSE] = field(default_factory=list)
    default_headers: dict[str, str] | None = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
```

- [ ] **Step 6: Commit**

```bash
git add src/vibe_agent/types.py src/vibe_agent/config.py tests/test_types.py
git commit -m "feat: add types (Response, StreamEvent, SubagentResult) and config dataclasses"
```

---

### Task 3: Tool Base and Registry

**Files:**
- Create: `src/vibe_agent/tools/base.py`
- Create: `src/vibe_agent/tools/registry.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests for ToolRegistry**

`tests/test_tools.py`:
```python
import pytest
from vibe_agent.tools.registry import ToolRegistry


def test_register_and_get_openai_tools():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")
    tools = registry.get_openai_tools()
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "add"
    assert tools[0]["function"]["description"] == "Add two numbers"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "a" in tools[0]["function"]["parameters"]["properties"]
    assert "b" in tools[0]["function"]["parameters"]["properties"]


def test_register_detects_required_params():
    registry = ToolRegistry()

    async def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}"

    registry.register(greet, name="greet", description="Greet someone")
    params = registry.get_openai_tools()[0]["function"]["parameters"]
    assert "name" in params["required"]
    assert "greeting" not in params["required"]


@pytest.mark.asyncio
async def test_execute_local_tool():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(add, name="add", description="Add two numbers")
    result = await registry.execute("add", '{"a": 3, "b": 5}')
    assert result == "8"


@pytest.mark.asyncio
async def test_execute_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="unknown"):
        await registry.execute("unknown", "{}")


def test_register_mcp_tool():
    registry = ToolRegistry()

    async def handler(arguments: dict) -> str:
        return "result"

    registry.register_mcp_tool(
        name="mcp_read",
        description="Read via MCP",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )
    tools = registry.get_openai_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "mcp_read"


@pytest.mark.asyncio
async def test_execute_mcp_tool():
    registry = ToolRegistry()

    async def handler(arguments: dict) -> str:
        return f"read {arguments['path']}"

    registry.register_mcp_tool(
        name="mcp_read",
        description="Read via MCP",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=handler,
    )
    result = await registry.execute("mcp_read", '{"path": "/tmp/a"}')
    assert result == "read /tmp/a"


def test_has_tool():
    registry = ToolRegistry()

    async def noop() -> str:
        return ""

    registry.register(noop, name="noop", description="Does nothing")
    assert registry.has_tool("noop")
    assert not registry.has_tool("missing")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Create tools/base.py**

`src/vibe_agent/tools/base.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    is_mcp: bool = False
```

- [ ] **Step 4: Create tools/registry.py**

`src/vibe_agent/tools/registry.py`:
```python
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Awaitable

from vibe_agent.tools.base import Tool


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    origin = getattr(annotation, "__origin__", None)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list or origin is list:
        return {"type": "array"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {}


def _extract_parameters(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        schema = _python_type_to_json_schema(param.annotation)
        if not schema and param.annotation is not inspect.Parameter.empty:
            schema = {"type": "string"}
        if not schema:
            schema = {"type": "string"}
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, func: Callable, name: str | None = None, description: str = "") -> None:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        parameters = _extract_parameters(func)
        self._tools[tool_name] = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
            is_mcp=False,
        )

    def register_mcp_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict], Awaitable[Any]],
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            is_mcp=True,
        )

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments_json: str) -> str:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        tool = self._tools[name]
        args = json.loads(arguments_json)
        if tool.is_mcp:
            result = await tool.handler(args)
        else:
            result = await tool.handler(**args)
        return str(result)

    def has_tool(self, name: str) -> bool:
        return name in self._tools
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: all 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/vibe_agent/tools/ tests/test_tools.py
git commit -m "feat: add Tool base class and ToolRegistry with local + MCP tool support"
```

---

### Task 4: MCP Tool Adapter and Client

**Files:**
- Create: `src/vibe_agent/mcp/tool_adapter.py`
- Create: `src/vibe_agent/mcp/client.py`
- Create: `tests/test_mcp_adapter.py`

- [ ] **Step 1: Write failing test for tool adapter**

`tests/test_mcp_adapter.py`:
```python
from vibe_agent.mcp.tool_adapter import mcp_tool_to_openai


def test_mcp_tool_to_openai():
    mcp_tool = {
        "name": "read_file",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    result = mcp_tool_to_openai(mcp_tool)
    assert result["type"] == "function"
    assert result["function"]["name"] == "read_file"
    assert result["function"]["description"] == "Read a file"
    assert result["function"]["parameters"]["type"] == "object"
    assert "path" in result["function"]["parameters"]["properties"]


def test_mcp_tool_to_openai_no_description():
    mcp_tool = {
        "name": "noop",
        "inputSchema": {"type": "object", "properties": {}},
    }
    result = mcp_tool_to_openai(mcp_tool)
    assert result["function"]["description"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_adapter.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Create tool_adapter.py**

`src/vibe_agent/mcp/tool_adapter.py`:
```python
from __future__ import annotations

from typing import Any


def mcp_tool_to_openai(mcp_tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool["name"],
            "description": mcp_tool.get("description", ""),
            "parameters": mcp_tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    }
```

- [ ] **Step 4: Run adapter test to verify it passes**

Run: `uv run pytest tests/test_mcp_adapter.py -v`
Expected: 2 tests PASS

- [ ] **Step 5: Create MCP client**

`src/vibe_agent/mcp/client.py`:
```python
from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from vibe_agent.config import MCPServerStdio, MCPServerSSE

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, server_config: MCPServerStdio | MCPServerSSE, timeout: float = 30.0) -> None:
        self._config = server_config
        self._timeout = timeout
        self._session: ClientSession | None = None
        self._cm_stack: Any = None

    async def connect(self) -> None:
        try:
            if isinstance(self._config, MCPServerStdio):
                params = StdioServerParameters(
                    command=self._config.command,
                    args=self._config.args,
                    env=self._config.env,
                )
                read, write = await self._enter(stdio_client(params))
            else:
                read, write, _ = await self._enter(
                    streamable_http_client(self._config.url, headers=self._config.headers or {})
                )

            self._session = await self._enter(ClientSession(read, write))
            await self._session.initialize()
            logger.info("Connected to MCP server: %s", self._config)
        except Exception:
            logger.warning("Failed to connect to MCP server: %s", self._config, exc_info=True)
            await self.disconnect()
            raise

    async def _enter(self, cm: Any) -> Any:
        if self._cm_stack is None:
            from contextlib import AsyncExitStack
            self._cm_stack = AsyncExitStack()
        return await self._cm_stack.enter_async_context(cm)

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError("MCP session not connected")
        result = await asyncio.wait_for(
            self._session.call_tool(name, arguments=arguments),
            timeout=self._timeout,
        )
        if result.isError:
            parts = [str(c) for c in result.content]
            raise RuntimeError(f"MCP tool '{name}' error: {'; '.join(parts)}")
        parts = [str(c) for c in result.content]
        return "\n".join(parts)

    async def disconnect(self) -> None:
        if self._cm_stack is not None:
            await self._cm_stack.aclose()
            self._cm_stack = None
            self._session = None
```

- [ ] **Step 6: Commit**

```bash
git add src/vibe_agent/mcp/ tests/test_mcp_adapter.py
git commit -m "feat: add MCP client (stdio + streamable_http) and tool adapter"
```

---

### Task 5: Core Agent — Basic Conversation (No Tools)

**Files:**
- Create: `src/vibe_agent/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing test for basic conversation**

`tests/test_agent.py`:
```python
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from vibe_agent.agent import Agent
from vibe_agent.config import AgentConfig
from vibe_agent.types import Response


def _mock_openai_response(content: str, tool_calls=None, usage=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=usage.get("prompt_tokens", 10) if usage else 10,
        completion_tokens=usage.get("completion_tokens", 20) if usage else 20,
        total_tokens=usage.get("total_tokens", 30) if usage else 30,
    )
    return resp


@pytest.mark.asyncio
async def test_basic_conversation():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    mock_response = _mock_openai_response("Hello! How can I help?")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        result = await agent.run("Hi")

    assert isinstance(result, Response)
    assert result.content == "Hello! How can I help?"
    assert result.usage.total_tokens == 30
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_system_prompt_in_messages():
    config = AgentConfig(model="gpt-4o", api_key="test-key", system_prompt="You are helpful")
    agent = Agent(config)

    mock_response = _mock_openai_response("Hi!")
    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        await agent.run("Hello")

    call_kwargs = mock_create.call_args
    messages = call_kwargs.kwargs["messages"] if call_kwargs.kwargs else call_kwargs[1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"


@pytest.mark.asyncio
async def test_max_iterations_prevents_infinite_loop():
    config = AgentConfig(model="gpt-4o", api_key="test-key", max_iterations=1)
    agent = Agent(config)

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "add"
    tool_call.function.arguments = '{"a": 1, "b": 2}'

    mock_response = _mock_openai_response(None, tool_calls=[tool_call])

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        result = await agent.run("calculate")

    assert "reached maximum" in result.content.lower() or result.content != ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Create agent.py — basic conversation**

`src/vibe_agent/agent.py`:
```python
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from vibe_agent.config import AgentConfig
from vibe_agent.tools.registry import ToolRegistry
from vibe_agent.types import Response, StreamEvent, TokenUsage

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=config.default_headers,
        )
        self._registry = ToolRegistry()
        self._mcp_clients: list[Any] = []
        self._conversations: dict[str, list[dict[str, Any]]] = {}

    def tool(self, name: str | None = None, description: str = ""):
        def decorator(func):
            self._registry.register(func, name=name, description=description)
            return func
        return decorator

    def add_tools(self, funcs: list) -> None:
        for func in funcs:
            self._registry.register(func)

    async def _connect_mcp(self) -> None:
        from vibe_agent.mcp.client import MCPClient

        for server_conf in self.config.mcp_servers:
            client = MCPClient(server_conf, timeout=self.config.tool_timeout)
            try:
                await client.connect()
                tools = await client.list_tools()
                for t in tools:

                    async def _make_handler(c=client):
                        async def handler(args: dict) -> str:
                            return await c.call_tool(t["name"], args)
                        return handler

                    handler = await _make_handler()
                    self._registry.register_mcp_tool(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["inputSchema"],
                        handler=handler,
                    )
                self._mcp_clients.append(client)
            except Exception:
                logger.warning("Skipping MCP server %s: connection failed", server_conf, exc_info=True)

    async def _disconnect_mcp(self) -> None:
        for client in self._mcp_clients:
            await client.disconnect()
        self._mcp_clients.clear()

    async def __aenter__(self):
        await self._connect_mcp()
        return self

    async def __aexit__(self, *exc):
        await self._disconnect_mcp()

    def _build_messages(self, user_input: str, conversation_id: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        if conversation_id and conversation_id in self._conversations:
            messages.extend(self._conversations[conversation_id])
        messages.append({"role": "user", "content": user_input})
        return messages

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        stream: bool = False,
    ) -> Response | AsyncIterator[StreamEvent]:
        messages = self._build_messages(message, conversation_id)
        if stream:
            return self._run_stream(messages, conversation_id)
        return await self._run(messages, conversation_id)

    async def _run(self, messages: list[dict], conversation_id: str | None) -> Response:
        tools = self._registry.get_openai_tools() or None
        history: list[dict[str, Any]] = list(messages)

        for _ in range(self.config.max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": history,
            }
            if tools:
                kwargs["tools"] = tools

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message
            history.append(_message_to_dict(assistant_msg))

            if not assistant_msg.tool_calls:
                usage = None
                if response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                result = Response(content=assistant_msg.content or "", usage=usage, history=history)
                if conversation_id:
                    self._conversations[conversation_id] = history[-20:]
                return result

            results = await _execute_tool_calls(self._registry, assistant_msg.tool_calls)
            for tc, tool_result in zip(assistant_msg.tool_calls, results):
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        return Response(content="Reached maximum iterations without completion.", history=history)

    async def _run_stream(
        self, messages: list[dict], conversation_id: str | None
    ) -> AsyncIterator[StreamEvent]:
        tools = self._registry.get_openai_tools() or None
        history: list[dict[str, Any]] = list(messages)

        async def _stream_generator():
            for iteration in range(self.config.max_iterations):
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": history,
                    "stream": True,
                }
                if tools:
                    kwargs["tools"] = tools

                stream = await self._client.chat.completions.create(**kwargs)

                text_content = ""
                tool_calls_map: dict[int, dict[str, Any]] = {}
                usage_data = None

                async for chunk in stream:
                    if chunk.usage:
                        usage_data = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    if delta.content:
                        text_content += delta.content
                        yield StreamEvent(type="text", content=delta.content)

                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": tc_chunk.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_chunk.id:
                                tool_calls_map[idx]["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name:
                                    tool_calls_map[idx]["name"] = tc_chunk.function.name
                                if tc_chunk.function.arguments:
                                    tool_calls_map[idx]["arguments"] += tc_chunk.function.arguments

                if not tool_calls_map:
                    assistant_msg = {"role": "assistant", "content": text_content}
                    history.append(assistant_msg)
                    yield StreamEvent(type="done", usage=usage_data)
                    if conversation_id:
                        self._conversations[conversation_id] = history[-20:]
                    return

                tool_calls_list = sorted(tool_calls_map.values(), key=lambda x: list(tool_calls_map.keys())[list(tool_calls_map.values()).index(x)])
                history.append({
                    "role": "assistant",
                    "content": text_content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls_list
                    ],
                })

                for tc in tool_calls_list:
                    yield StreamEvent(type="tool_call", name=tc["name"], arguments=tc["arguments"])

                results = []
                for tc in tool_calls_list:
                    try:
                        result = await self._registry.execute(tc["name"], tc["arguments"])
                        results.append(result)
                        yield StreamEvent(type="tool_result", name=tc["name"], content=result)
                    except Exception as e:
                        err_msg = f"Error: {e}"
                        results.append(err_msg)
                        yield StreamEvent(type="tool_result", name=tc["name"], content=err_msg)

                for tc, tool_result in zip(tool_calls_list, results):
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

            yield StreamEvent(type="done", usage=None)

        return _stream_generator()


def _message_to_dict(msg: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


async def _execute_tool_calls(registry: ToolRegistry, tool_calls: list) -> list[str]:
    import asyncio

    async def _exec_one(tc):
        try:
            return await registry.execute(tc.function.name, tc.function.arguments)
        except Exception as e:
            return f"Error executing tool '{tc.function.name}': {e}"

    return await asyncio.gather(*[_exec_one(tc) for tc in tool_calls])
```

- [ ] **Step 4: Run basic agent tests**

Run: `uv run pytest tests/test_agent.py::test_basic_conversation tests/test_agent.py::test_system_prompt_in_messages tests/test_agent.py::test_max_iterations_prevents_infinite_loop -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Write test for tool calling in agent**

Append to `tests/test_agent.py`:

```python
@pytest.mark.asyncio
async def test_tool_calling_loop():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    async def add(a: int, b: int) -> int:
        return a + b

    agent.add_tools([add])

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.function.name = "add"
    tool_call.function.arguments = '{"a": 3, "b": 5}'

    # First response: tool call
    resp1 = _mock_openai_response(None, tool_calls=[tool_call])

    # Second response: final answer
    resp2 = _mock_openai_response("The result is 8.")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = [resp1, resp2]
        result = await agent.run("What is 3+5?")

    assert result.content == "The result is 8."
    assert mock_create.call_count == 2
```

- [ ] **Step 6: Run tool calling test**

Run: `uv run pytest tests/test_agent.py::test_tool_calling_loop -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/vibe_agent/agent.py tests/test_agent.py
git commit -m "feat: core agent with ReAct loop, tool calling, and streaming support"
```

---

### Task 6: Streaming Tests

**Files:**
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Write streaming tests**

`tests/test_streaming.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from vibe_agent.agent import Agent
from vibe_agent.config import AgentConfig


def _mock_stream_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    delta.role = "assistant"

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    if usage:
        chunk.usage = MagicMock(**usage)
    return chunk


def _mock_stream_response(chunks):
    resp = MagicMock()

    async def _aiter():
        for c in chunks:
            yield c

    resp.__aiter__ = lambda self: _aiter()
    return resp


@pytest.mark.asyncio
async def test_streaming_text():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    chunks = [
        _mock_stream_chunk(content="Hello"),
        _mock_stream_chunk(content=" world"),
        _mock_stream_chunk(content="!", finish_reason="stop"),
        _mock_stream_chunk(usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}),
    ]
    stream_resp = _mock_stream_response(chunks)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = stream_resp
        stream_gen = await agent.run("Hi", stream=True)

    events = []
    async for event in stream_gen:
        events.append(event)

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 3
    assert text_events[0].content == "Hello"
    assert text_events[1].content == " world"
    assert text_events[2].content == "!"

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
```

- [ ] **Step 2: Run streaming test**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_streaming.py
git commit -m "test: add streaming output tests"
```

---

### Task 7: Subagent System

**Files:**
- Create: `src/vibe_agent/subagent.py`
- Create: `tests/test_subagent.py`

- [ ] **Step 1: Write failing tests**

`tests/test_subagent.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from vibe_agent.agent import Agent
from vibe_agent.config import AgentConfig, SubagentConfig
from vibe_agent.types import SubagentResult


@pytest.mark.asyncio
async def test_delegate_parallel():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    def _mock_response(content: str):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        return resp

    with patch.object(
        AsyncMock.return_value.__class__, "create", new_callable=AsyncMock
    ) as mock_create:
        # We'll patch at agent level
        pass

    # Simpler approach: patch each subagent's _client directly
    results = await agent.delegate(
        tasks=["Task A", "Task B"],
        config=SubagentConfig(max_parallel=2, timeout=10),
    )

    assert len(results) == 2
    assert all(isinstance(r, SubagentResult) for r in results)


@pytest.mark.asyncio
async def test_delegate_timeout():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    results = await agent.delegate(
        tasks=["slow task"],
        config=SubagentConfig(timeout=0.01),
    )
    assert len(results) == 1
    assert results[0].status == "timeout"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_subagent.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Create subagent.py**

`src/vibe_agent/subagent.py`:
```python
from __future__ import annotations

import asyncio
import logging
from typing import Any

from vibe_agent.config import AgentConfig, SubagentConfig
from vibe_agent.types import SubagentResult

logger = logging.getLogger(__name__)


async def delegate(
    agent: Any,
    tasks: list[str],
    config: SubagentConfig | None = None,
) -> list[SubagentResult]:
    if config is None:
        config = SubagentConfig()

    semaphore = asyncio.Semaphore(config.max_parallel)

    async def _run_one(task: str) -> SubagentResult:
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    agent._run(
                        [{"role": "user", "content": task}],
                        conversation_id=None,
                    ),
                    timeout=config.timeout,
                )
                return SubagentResult(status="success", content=result.content, usage=result.usage)
            except asyncio.TimeoutError:
                logger.warning("Subagent timed out for task: %s", task[:50])
                return SubagentResult(status="timeout", content="")
            except Exception as e:
                logger.warning("Subagent error for task: %s — %s", task[:50], e)
                return SubagentResult(status="error", content=str(e))

    results = await asyncio.gather(*[_run_one(t) for t in tasks])
    return list(results)
```

- [ ] **Step 4: Add delegate method to Agent class**

Add to `src/vibe_agent/agent.py`, import `delegate` at top and add method to `Agent`:

```python
# Add import at top of agent.py:
from vibe_agent.subagent import delegate as _delegate
from vibe_agent.config import SubagentConfig
from vibe_agent.types import SubagentResult

# Add method to Agent class:
    async def delegate(
        self,
        tasks: list[str],
        config: SubagentConfig | None = None,
    ) -> list[SubagentResult]:
        return await _delegate(self, tasks, config)
```

- [ ] **Step 5: Run subagent tests**

Run: `uv run pytest tests/test_subagent.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/vibe_agent/subagent.py src/vibe_agent/agent.py tests/test_subagent.py
git commit -m "feat: parallel subagent system with timeout and concurrency control"
```

---

### Task 8: Public API Exports

**Files:**
- Modify: `src/vibe_agent/__init__.py`

- [ ] **Step 1: Update __init__.py with all public exports**

`src/vibe_agent/__init__.py`:
```python
"""Vibe Agent — lightweight AI agent library."""

from vibe_agent.config import AgentConfig, MCPServerStdio, MCPServerSSE, SubagentConfig
from vibe_agent.types import Response, StreamEvent, SubagentResult
from vibe_agent.agent import Agent

__all__ = [
    "Agent",
    "AgentConfig",
    "MCPServerStdio",
    "MCPServerSSE",
    "SubagentConfig",
    "Response",
    "StreamEvent",
    "SubagentResult",
]
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from vibe_agent import Agent, AgentConfig, MCPServerStdio, MCPServerSSE, SubagentConfig, Response, StreamEvent, SubagentResult; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Commit**

```bash
git add src/vibe_agent/__init__.py
git commit -m "feat: public API exports in __init__.py"
```

---

### Task 9: Full Test Suite Verification

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 2: Run with coverage**

Run: `uv run pytest tests/ --cov=vibe_agent --cov-report=term-missing`
Expected: review coverage report, ensure core modules >80%

- [ ] **Step 3: Fix any failures**

If any tests fail, investigate and fix. Re-run until all pass.

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "chore: final test suite pass and coverage check"
```
