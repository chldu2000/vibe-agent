from air_agent.mcp.tool_adapter import mcp_tool_to_openai


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
