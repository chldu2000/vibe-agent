FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir air-agent

COPY agent-config.json /app/agent-config.json
COPY run_agent.py /app/run_agent.py

ENV AIR_CONFIG=/app/agent-config.json

CMD ["python", "/app/run_agent.py"]
