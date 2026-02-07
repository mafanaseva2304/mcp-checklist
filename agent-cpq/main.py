import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

VLLM_API_BASE = "http://10.0.1.1:10000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "nvidia/Nemotron-Orchestrator-8B"

llm = ChatOpenAI(
    openai_api_base=VLLM_API_BASE,
    openai_api_key=VLLM_API_KEY,
    model_name=MODEL_NAME,
)

SYSTEM_PROMPT = """
Отвечай только на русском языке.
Ты — эксперт-аудитор ML-моделей в системе контроль и прогноз какчества (КПК).
Большая часть моделей - виртуальные анализаторы качества либо модели смешения.
Твоя задача — выявлять проблемы моделей по запросам пользователя: отсутствие статусов/тегов/значений/дат, отключённые модели, несоответствия конфигов (IPK/train), пределов (tech_min/max), предикторов, бинарников и данных (за час/последние).


Правила:
1. Для запроса о проблемах — вызови **все релевантные** инструменты (цепочкой, если нужно).
2. Собери результаты, проанализируй: сгруппируй по типам проблем, посчитай модели, выдели критику (отключённые, без данных).
3. Ответ: Markdown-таблица с колонками "Проблема | Инструмент | Кол-во моделей | Примеры моделей".
4. Если нет проблем — скажи "OK".
5. Только факты из инструментов, без домыслов.
6. Финал: "Рекомендации: [кратко, напр. обновить статусы]".
"""

client = MultiServerMCPClient(
    {
        "": {
            "transport": "http",
            "url": "http://10.0.1.1:8002/mcp",
        }
    }
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


async def main():
    # Вариант 1: просто взять все инструменты с одного сервера
    TOOLS = await client.get_tools(server_name="")  # <-- ВОТ ТАК ПРАВИЛЬНО [web:195][web:196]

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Выведи все модели, которые на текущий момент упираются в максимальные или минимальные пределы.",
                }
            ]
        }
    )

    lol = str(result["messages"][-1])
    only_text = lol.split("'")[1].split("\\n")
    print(lol)


if __name__ == "__main__":
    asyncio.run(main())