import asyncio
from fastmcp import Client
import lmstudio as lms
import json

# Создаём MCP-клиент для удалённого MCP-сервера
MCP_SERVER_URL = "http://localhost:8000/mcp"
client = Client(MCP_SERVER_URL, timeout=30.0)  # Можно добавить обработчики логов и ошибки

# Асинхронные функции-прокси для всех MCP инструментов
async def get_mcp_tools(client):
    tools = await client.list_tools()  # Получаем описание инструментов с сервера

    proxy_funcs = []
    for tool in tools:
        tool_name = tool.name
        tool_desc = tool.description or "Given a tool argument, returns answer MCP tool."
        parameters = tool.inputSchema.properties if hasattr(tool.inputSchema, "properties") else {}

        # Динамически создаём функцию под каждый инструмент
        def proxy_factory(name, desc, params):
            async def tool_proxy(**kwargs):
                # Вызов инструмента через fastmcp
                # print(kwargs)
                result = await client.call_tool(name, arguments=kwargs)
                # Возвращаем text-результат для LLM
                # print(result)
                try:
                    res_json = json.loads(result.content[0].text)
                    res = f'"result":[{res_json["result"][:10]}]'
                except Exception as ex:
                    res = str(result.text if hasattr(result, 'text') else result)
                    print(ex)
                return res
            tool_proxy.__name__ = name
            tool_proxy.__doc__ = desc
            tool_proxy.__annotations__ = {param: str for param in params}
            tool_proxy.__annotations__["return"] = str
            return tool_proxy

        proxy_funcs.append(proxy_factory(tool_name, tool_desc, parameters))
    return proxy_funcs

def print_fragment(fragment, round_index=0):
    # .act() supplies the round index as the second parameter
    # Setting a default value means the callback is also
    # compatible with .complete() and .respond().
    print(fragment.content, end="", flush=True)
    
# Инициализация и запуск LM Studio с этими инструментами
async def main():
    # Инициализация LM Studio модели и чата
    async with lms.AsyncClient("192.168.10.21:8591") as llmclient:
        model = await llmclient.llm.model("openai/gpt-oss-20b")
        chat = lms.Chat("Ты — ассистент с удалёнными инструментами")

        # Инициализация MCP клиента и создание прокси-функций
        async with client:
            proxy_tools = await get_mcp_tools(client)
            
            while True:
                try:
                    user_input = input("Вопрос к llm: ")
                except EOFError:
                    print()
                    break
                if not user_input:
                    break
                chat.add_user_message(user_input)
                print("Bot: ", end="", flush=True)
                response = await model.act(
                chat,
                proxy_tools,
                on_message=chat.append,
                on_prediction_fragment=print_fragment
                )
                print()
            

            # # Пример диалога.
            # chat.add_user_message("Есть ли ошибки в логах таски smart_mpc_read в последних запусках?")

            # # Вызов .act() с инструментами. Всё остальное - как обычно!
            # response = await model.act(
            #     chat,
            #     proxy_tools,
            #     on_message=chat.append,
            #     on_prediction_fragment=print_fragment
            # )
            # print(response)

if __name__ == "__main__":
    asyncio.run(main())