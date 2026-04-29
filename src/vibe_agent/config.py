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
