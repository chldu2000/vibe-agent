from __future__ import annotations

import asyncio
from typing import Any

from air_agent import Agent, AgentConfig, LLMResponse


class ExampleProvider:
    supports_tools = True
    supports_streaming = False

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools=None,
        **options: Any,
    ) -> LLMResponse:
        return LLMResponse(content=f"example response from {model}")

    async def stream(self, **kwargs: Any):
        if False:
            yield


async def main() -> str:
    agent = Agent(AgentConfig(model="example-model", provider=ExampleProvider()))
    response = await agent.run("hello")
    return response.content


if __name__ == "__main__":
    print(asyncio.run(main()))
