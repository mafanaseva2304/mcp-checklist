import asyncio
from fastmcp import Client, FastMCP
import lmstudio as lms
import json
# In-memory server (ideal for testing)
# server = FastMCP("TestServer")
# client = Client(server)
config = {
    "mcpServers": {
        "server_name": {
            # Remote HTTP/SSE server
            "transport": "http",  # or "sse" 
            "url": "http://localhost:8000/mcp/",
            "headers": {"accept": "application/json", "X-Custom-Header": "value"}
        }
    }
}
client = Client(config)
# HTTP server
# client = Client("http://192.168.10.50:8000/mcp/", message_handler = "accept": "application/json",)

async def main():
    async with client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        for tool in tools:
            # if 'tag' not in tool.name or 'list' not in tool.name:
            #     if 'search' not in tool.name:
            #         continue
            print(tool.name,tool.annotations)
        print(resources)
        print(prompts)
        print()
        result = await client.call_tool("find_models_without_ipk_metrics", {})
        # try:
        #     res_json = json.loads(result.content[0].text)
        #     res = f'"result":[{res_json["result"][:10]}]'
        # except Exception as ex:
        #     res = result
        #     print(ex)
        print(result)
        print()
        print()
        print()
        # # Execute operations
        # result = await client.call_tool("periodic_tasks_api_task_manager_periodic_tasks")
        # print(result) {"prompt": "вызови команду mcp.tools: periodic_tasks_api_task_manager_periodic_tasks и на основе ее вывода составь список запущенных тасок"}
        # with lms.Client("192.168.10.21:8591") as llmclient:
        #     #   Какие типы и виды моделей и предобработки данных есть в приложении ситнеза моделей?
        #     model = llmclient.llm.model("openai/gpt-oss-20b")
        #     # messages = [
        #     # {"role": "system", "content": system_prompt},
        #     # {"role": "user", "content": prompt}  # где prompt — ваш пользовательский запрос
        #     # ]
            
        #     # chat = lms.Chat.from_history({"messages": messages})
        #     chat = lms.Chat("Отвечай на русском. Какие таски есть в таск менеджере?")
        #     ract = model.act(
        #         chat,
        #         [client.call_tool],
        #         on_message=print,
        #         )
        #     # ret = model.respond(chat)
        #     print(ract)
        
        # result = await client.call_tool("agent_by_model", {"prompt": "Отвечай на русском. Какие таски есть в таск менеджере?"})
        # print(result)
        

asyncio.run(main())


# sttt = '''<tool_call>
# {
#   "name": "periodic_tasks_api_task_manager_periodic_tasks",
#   "arguments": {}
# }
# </tool_call>'''
# print(json.loads(sttt.replace('<tool_call>','').replace('</tool_call>','')))