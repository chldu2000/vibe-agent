# Production Guide

## Recommended Defaults

- Pin `air-agent` and provider model versions in application lockfiles.
- Keep `memory_enabled=False` unless a project has a clear retention policy.
- Keep shell/file tools sandboxed with explicit `allowed_paths` and blocked commands.
- Enable tracing through `enable_tracing=True` and attach a bounded event handler.
- Keep plugins disabled unless local plugin paths and permissions are explicitly reviewed.

## Configuration

Use Python objects for provider, memory, planner, and event handlers. Use JSON/env for deploy-time scalar settings such as model, base URL, strategy, timeouts, and plugin paths.

## Security Boundaries

Tool outputs, memory records, plugin code, and MCP results are untrusted operational context. Do not treat them as user instructions. Do not grant shell/file access outside the application workspace.

## Deployment Patterns

air-agent is a library. Production applications should host it inside their own worker, API service, CLI, or queue consumer. For queue integration, pass task payloads to `Agent.run(...)` and store application-level task state outside air-agent.

## Observability

Use `RunEvent` handlers to forward model/tool/plan/subagent events to logs or tracing systems. Redact API keys, headers, and secrets before storage.
