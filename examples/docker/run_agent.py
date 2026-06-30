import asyncio
import os

from air_agent import Agent, AgentConfig


async def main() -> None:
    config_path = os.environ.get("AIR_CONFIG", "agent-config.json")
    agent = Agent(AgentConfig.from_json(config_path))
    response = await agent.run("Say hello from air-agent.")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
