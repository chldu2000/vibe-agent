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
