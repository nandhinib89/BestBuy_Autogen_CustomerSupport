import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agents.order_agent import create_order_agent

load_dotenv()


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_order_agent(model_client)

    question = "My order was cancelled but I was still charged. What happened?"

    result = await agent.run(task=question)

    print("\n--- ALL MESSAGES ---\n")

    for i, message in enumerate(result.messages):
        print(f"\nMESSAGE {i}")
        print(f"TYPE: {type(message).__name__}")
        print(f"CONTENT:\n{message.content}")
        print("-" * 80)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())