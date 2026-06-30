# API Stability And Semver Policy

air-agent v1.0 follows semantic versioning for Stable public APIs.

## Stable In v1.0

- `Agent`
- `AgentConfig`
- `Response`, `StreamEvent`, `RunEvent`
- Local tool registration: `Agent.tool(...)`, `Agent.add_tools(...)`
- Built-in tool configuration: `BuiltinToolsConfig`
- MCP server configuration: `MCPServerStdio`, `MCPServerSSE`
- Provider protocol and response types: `LLMProvider`, `LLMResponse`, `LLMStreamChunk`, `LLMToolCall`
- Memory protocol and built-ins: `MemoryStore`, `MemoryRecord`, `InMemoryMemoryStore`, `FileMemoryStore`
- Planner protocol and types: `Planner`, `LLMPlanner`, `Plan`, `PlanStep`, `StepResult`
- Public errors from `air_agent.errors`

## Experimental In v1.0

- Local plugin registry: `PluginManifest`, `PluginContext`, plugin entrypoint loading
- Multi-agent aggregation modes beyond basic `delegate(tasks)`
- Legacy skill router types: `SkillRouter`, `LLMSkillRouter`, `SkillRouteResult`
- Future checkpoint/resume boundaries

Experimental APIs may change in minor releases with migration notes.

## Compatibility Promise

- Patch releases fix bugs and documentation without changing public behavior.
- Minor releases may add optional fields, methods, events, and config options.
- Breaking changes to Stable APIs require a major version.
- Experimental APIs may change before v2.0, but changes must be documented.
