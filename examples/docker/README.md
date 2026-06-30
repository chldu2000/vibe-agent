# Docker Example

Build the image:

```bash
docker build -t air-agent-example .
```

Run it with an OpenAI API key:

```bash
docker run --rm -e OPENAI_API_KEY="$OPENAI_API_KEY" air-agent-example
```

The container loads `/app/agent-config.json` through `AgentConfig.from_json(...)`. Override the config path by setting `AIR_CONFIG`.
