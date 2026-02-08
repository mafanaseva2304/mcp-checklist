import asyncio

from fastmcp import FastMCP
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from settings import settings

llm = ChatOpenAI(
    openai_api_base=settings.openai.api_base,
    openai_api_key=settings.openai.api_key,
    model_name=settings.openai.model_name,
    temperature=settings.openai.temperature,
    max_tokens=settings.openai.max_tokens,
    streaming=settings.openai.streaming,
    max_retries=settings.openai.max_retries,
    request_timeout=settings.openai.request_timeout,
)

mcp_client = MultiServerMCPClient(
    {
        name: {"url": srv.url, "transport": srv.transport}
        for name, srv in settings.mcp_servers.items()
    }
)

mcp = FastMCP(settings.mcp_self.name)


@mcp.tool(
    name=settings.mcp_self.tool_name,
    description=settings.mcp_self.tool_description,
)
async def _agent_tool(query: str) -> str:
    server_name = list(settings.mcp_servers.keys())[0]
    tools = await mcp_client.get_tools(server_name=server_name)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=settings.system_prompt,
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = (
            last.get("content")
            if isinstance(last, dict)
            else getattr(last, "content", None)
        )
        if content is not None:
            return content
    return str(result)


if __name__ == "__main__":
    mcp.run(
        transport=settings.mcp_self.transport,
        host=settings.mcp_self.host,
        port=settings.mcp_self.port,
    )
