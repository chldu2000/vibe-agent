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
