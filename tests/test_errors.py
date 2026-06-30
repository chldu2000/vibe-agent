import pytest

from air_agent import (
    Agent,
    AgentConfig,
    AirAgentError,
    ConfigurationError,
    LLMResponse,
    MCPConnectionError,
    MemoryError,
    PlannerError,
    PluginLoadError,
    ProviderError,
    ToolExecutionError,
    ToolPermissionError,
)
from air_agent.planner import LLMPlanner, PlanContext
from air_agent.tools.builtin import PermissionDeniedError
from air_agent.tools.registry import ToolRegistry


def test_public_error_hierarchy_is_exported():
    assert issubclass(ConfigurationError, AirAgentError)
    assert issubclass(ProviderError, AirAgentError)
    assert issubclass(ToolExecutionError, AirAgentError)
    assert issubclass(ToolPermissionError, ToolExecutionError)
    assert issubclass(ToolPermissionError, PermissionError)
    assert issubclass(MCPConnectionError, AirAgentError)
    assert issubclass(PluginLoadError, AirAgentError)
    assert issubclass(MemoryError, AirAgentError)
    assert issubclass(PlannerError, AirAgentError)


def test_permission_denied_error_remains_backward_compatible():
    assert issubclass(PermissionDeniedError, ToolPermissionError)
    with pytest.raises(PermissionError):
        raise PermissionDeniedError("denied")


def test_invalid_strategy_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="strategy"):
        AgentConfig(strategy="bad")  # type: ignore[arg-type]


def test_unsupported_provider_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="Unsupported provider"):
        Agent(AgentConfig(provider="unknown"))


@pytest.mark.asyncio
async def test_provider_without_tool_support_raises_provider_error():
    class NoToolProvider:
        supports_tools = False
        supports_streaming = False

        async def complete(self, **kwargs):
            raise AssertionError("not called")

        async def stream(self, **kwargs):
            raise AssertionError("not called")

    agent = Agent(AgentConfig(model="fake", provider=NoToolProvider()))

    @agent.tool()
    async def add(a: int, b: int) -> int:
        return a + b

    with pytest.raises(ProviderError, match="does not support tool calling"):
        await agent.run("add")


@pytest.mark.asyncio
async def test_planner_invalid_json_raises_planner_error():
    class BadProvider:
        supports_tools = False
        supports_streaming = False

        async def complete(self, **kwargs):
            return LLMResponse(content="not-json")

        async def stream(self, **kwargs):
            raise AssertionError("not called")

    planner = LLMPlanner(provider=BadProvider(), model="fake")
    with pytest.raises(PlannerError, match="not valid JSON"):
        await planner.create_plan("goal", PlanContext(goal="goal", messages=[]))


@pytest.mark.asyncio
async def test_registry_execute_unknown_still_raises_key_error():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="missing"):
        await registry.execute("missing", "{}")
