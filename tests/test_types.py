import pytest
from air_agent.types import Response, StreamEvent, SubagentResult


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
    assert e.usage.total_tokens == 15


def test_subagent_result_success():
    r = SubagentResult(status="success", content="done")
    assert r.status == "success"
    assert r.content == "done"


def test_subagent_result_timeout():
    r = SubagentResult(status="timeout", content="")
    assert r.status == "timeout"
