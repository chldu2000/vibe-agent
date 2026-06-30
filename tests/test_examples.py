from pathlib import Path
import importlib.util

import pytest

from air_agent import Agent, AgentConfig, LLMResponse, LLMToolCall


def test_v1_docs_exist():
    for path in [
        "docs/api-stability.md",
        "docs/production.md",
        "docs/migration-v1.0.md",
        "docs/releases/v1.0.md",
    ]:
        assert Path(path).is_file()


@pytest.mark.asyncio
async def test_production_example_main_returns_content():
    spec = importlib.util.spec_from_file_location(
        "production_agent_example",
        Path("examples/production_agent.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert await module.main() == "example response from example-model"


@pytest.mark.asyncio
async def test_plugin_example_registers_namespaced_tool():
    class Provider:
        supports_tools = True
        supports_streaming = False

        def __init__(self):
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="tc_1",
                            name="example.echo",
                            arguments='{"text":"hello"}',
                        )
                    ],
                )
            return LLMResponse(content="done")

        async def stream(self, **kwargs):
            if False:
                yield

    agent = Agent(
        AgentConfig(
            model="fake",
            provider=Provider(),
            plugins=["examples/plugin_example"],
        )
    )
    response = await agent.run("echo hello")
    assert response.content == "done"
    assert any(message.get("content") == "hello" for message in response.history)
