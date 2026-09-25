import asyncio
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.product_agent import create_product_agent


load_dotenv()


async def main():

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_product_agent(model_client)

    question = "Is the Sony WH-1000XM6 currently in stock at a Best Buy Canada store?"

    result = await agent.run(task=question)

    print("\n--- PRODUCT AGENT RESPONSE ---\n")

    print(result.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())