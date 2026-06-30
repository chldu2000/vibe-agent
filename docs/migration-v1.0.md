# Migrating From v0.x To v1.0

## What Stays Compatible

- `Agent.run(...)` default ReAct behavior.
- `AgentConfig` scalar fields and JSON/env loading.
- Local tool registration with `agent.tool(...)` and `agent.add_tools(...)`.
- MCP server configuration.
- Provider, Memory, and Planner protocols.
- `delegate(tasks)` returning `list[SubagentResult]` when no aggregation is requested.

## What Changed

- Public framework errors are available from `air_agent.errors` and top-level `air_agent`.
- Plugin and advanced collaboration APIs are marked Experimental in v1.0.
- Skills are loaded explicitly through the `use_skill` tool; automatic skill-body injection is not the default behavior.

## Recommended Updates

- Catch `AirAgentError` for framework errors that should be handled uniformly.
- Catch specific errors such as `ProviderError`, `ToolPermissionError`, or `PluginLoadError` when recovery behavior differs.
- Review tool sandbox settings before production deployment.
- Use fake providers in tests instead of live model calls.
