from __future__ import annotations

import asyncio
import logging
from typing import Any

from air_agent.config import SubagentConfig
from air_agent.types import SubagentResult

logger = logging.getLogger(__name__)


async def delegate(
    agent: Any,
    tasks: list[str],
    config: SubagentConfig | None = None,
) -> list[SubagentResult]:
    if config is None:
        config = SubagentConfig()

    semaphore = asyncio.Semaphore(config.max_parallel)

    async def _run_one(task: str) -> SubagentResult:
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    agent._run(
                        [{"role": "user", "content": task}],
                        conversation_id=None,
                    ),
                    timeout=config.timeout,
                )
                return SubagentResult(status="success", content=result.content, usage=result.usage)
            except asyncio.TimeoutError:
                logger.warning("Subagent timed out for task: %s", task[:50])
                return SubagentResult(status="timeout", content="")
            except Exception as e:
                logger.warning("Subagent error for task: %s — %s", task[:50], e)
                return SubagentResult(status="error", content=str(e))

    results = await asyncio.gather(*[_run_one(t) for t in tasks])
    return list(results)
