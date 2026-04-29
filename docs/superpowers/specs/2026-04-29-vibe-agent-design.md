# Vibe Agent 设计文档

轻量级 Python AI Agent 库，支持 OpenAI tool calling 循环、MCP server 连接和并行 subagent。

## 项目结构

```
vibe-agent/
├── pyproject.toml
├── src/
│   └── vibe_agent/
│       ├── __init__.py        # 公开 API 导出
│       ├── agent.py           # 核心 agent loop
│       ├── config.py          # 配置数据类
│       ├── types.py           # 共享类型定义
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py    # 工具注册中心
│       │   └── base.py        # Tool 协议/基类
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── client.py      # MCP 客户端 (stdio + SSE)
│       │   └── tool_adapter.py # MCP tool → OpenAI tool 格式转换
│       └── subagent.py        # 并行 subagent 管理器
└── tests/
```

## 依赖

- `openai` — LLM 调用与 tool calling
- `mcp` — MCP 协议客户端（官方 Python SDK）
- `pydantic` — 配置与类型的校验

## 核心 Agent Loop

ReAct 循环：LLM 回复 → 检测 tool calls → 执行工具 → 结果回传 LLM → 重复，直到无 tool call。

1. 组装 messages（system_prompt + 历史 + 用户输入）
2. 调用 OpenAI Chat Completion API，传入 tools 参数
3. 如果 response 无 tool_calls → 返回最终回复
4. 如果有 tool_calls → 并行执行（`asyncio.gather`），将结果追加到 messages
5. 回到步骤 2，直到无 tool call 或达到 `max_iterations`（默认 20）

**Response 对象**包含：最终文本内容、token 用量、完整调用历史。

### 流式输出

`agent.run()` 支持 `stream=True`，返回 `AsyncIterator[StreamEvent]`：

```python
async for event in await agent.run("解释量子计算", stream=True):
    if event.type == "text":
        print(event.content, end="")          # 文本增量
    elif event.type == "tool_call":
        print(f"调用工具: {event.name}")       # 工具调用通知
    elif event.type == "tool_result":
        print(f"工具结果: {event.content}")    # 工具执行结果
    elif event.type == "done":
        print(f"\n完成，token: {event.usage}") # 最终汇总
```

- LLM 文本回复使用 OpenAI streaming API 逐 token 输出
- tool call 开始/结束时 emit 事件，调用方可以实时展示进度
- 非 streaming 模式（默认）仍返回完整 `Response` 对象

## 工具系统

两类工具统一注册到 `ToolRegistry`，对外暴露一致的 OpenAI tool JSON 格式。

### 本地工具

- 装饰器注册：`@agent.tool(name, description)`
- 也可批量添加：`agent.add_tools([fn1, fn2])`
- 自动从函数签名和 type hints 提取 parameters JSON Schema
- 必填/选填从默认值判断

### MCP 工具

- MCP server 连接后通过 `list_tools()` 发现工具
- `tool_adapter.py` 将 MCP tool schema 转换为 OpenAI `function` 格式
- 执行时委托给 `MCPClient.call_tool()`

### ToolRegistry 职责

- `register(fn, name, description)` — 注册本地函数
- `register_mcp_tools(tools)` — 注册 MCP 工具
- `get_openai_tools()` — 返回 OpenAI `tools` JSON
- `execute(name, arguments_json)` — 按名称执行，返回字符串结果

## MCP 客户端

支持 stdio 和 SSE/StreamableHTTP 两种 transport。

### 配置

```python
mcp_servers=[
    MCPServerStdio(command="npx", args=["-y", "@anthropic/mcp-server-filesystem", "/tmp"]),
    MCPServerSSE(url="http://localhost:8080/sse"),
]
```

### MCPClient 职责

- `connect()` — 建立 transport + 初始化 session
- `list_tools()` — 发现 server 暴露的工具
- `call_tool(name, arguments)` — 调用工具返回结果
- `disconnect()` — 关闭连接清理资源

### 生命周期

- Agent 启动时批量连接所有 MCP server
- 通过 `async with` 上下文管理器确保清理
- stdio 用 `mcp` SDK 的 `stdio_client`，SSE 用 `sse_client`
- 每个 MCP server 是独立的 `MCPClient` 实例

### 错误处理

- 单个 server 连接失败不阻塞其他 server
- tool call 超时（可配置，默认 30s）
- 连接断开返回明确错误信息，不崩溃

## Subagent 系统

并行执行型 subagent，主 agent 创建多个子 agent 并发运行。

### 使用方式

```python
results = await agent.delegate(
    tasks=["任务A", "任务B", "任务C"],
    config=SubagentConfig(max_parallel=2, timeout=60),
)
```

### 实现

- 每个 task 创建独立 `Agent` 实例（共享 LLM client 配置）
- `asyncio.gather` + `Semaphore` 控制并发上限
- 可选继承主 agent 的工具快照
- 单个 subagent 超时标记为 `SubagentResult(status="timeout")`，不影响其他

### 与主 Agent 的区别

- 无独立 MCP 连接（继承主 agent 工具快照）
- 独立 message 历史，不污染主对话
- 执行完毕即销毁，无状态残留

## 公开 API

### 核心导出

- `Agent`, `AgentConfig` — 主入口
- `MCPServerStdio`, `MCPServerSSE` — MCP 配置
- `SubagentConfig` — subagent 配置
- `Response`, `SubagentResult`, `StreamEvent` — 返回类型

### 使用示例

```python
from vibe_agent import Agent, AgentConfig, MCPServerStdio

# 基础对话
agent = Agent(AgentConfig(model="gpt-4o", api_key="sk-xxx"))
response = await agent.run("你好")

# 带 MCP server
agent = Agent(AgentConfig(
    model="gpt-4o",
    mcp_servers=[MCPServerStdio(command="npx", args=["..."])],
))
async with agent:
    response = await agent.run("列出文件")

# 注册本地工具
@agent.tool(description="计算两数之和")
async def add(a: int, b: int) -> int:
    return a + b

# 并行 subagent
results = await agent.delegate(tasks=["任务A", "任务B"])
```

### 安装与引用

`pip install vibe-agent` 或 `pip install -e .`，其他项目中直接 `from vibe_agent import ...`。
