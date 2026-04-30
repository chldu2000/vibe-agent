import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from air_agent.agent import Agent
from air_agent.config import AgentConfig, SubagentConfig
from air_agent.types import SubagentResult


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


@pytest.mark.asyncio
async def test_delegate_parallel():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _mock_response("done")
        results = await agent.delegate(
            tasks=["Task A", "Task B"],
            config=SubagentConfig(max_parallel=2, timeout=10),
        )

    assert len(results) == 2
    assert all(isinstance(r, SubagentResult) for r in results)
    assert all(r.status == "success" for r in results)


@pytest.mark.asyncio
async def test_delegate_timeout():
    config = AgentConfig(model="gpt-4o", api_key="test-key")
    agent = Agent(config)

    async def _slow_response(*args, **kwargs):
        import asyncio
        await asyncio.sleep(10)
        return _mock_response("done")

    with patch.object(agent._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = _slow_response
        results = await agent.delegate(
            tasks=["slow task"],
            config=SubagentConfig(timeout=0.01),
        )

    assert len(results) == 1
    assert results[0].status == "timeout"
