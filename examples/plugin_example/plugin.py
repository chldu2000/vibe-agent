def register(context):
    async def echo(text: str) -> str:
        return text

    context.register_tool(echo, namespace="example", description="Echo text for documentation examples")
