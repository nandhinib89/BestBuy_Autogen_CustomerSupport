import asyncio
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.availability_agent import create_availability_agent


load_dotenv()


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_availability_agent(model_client)

    task = "Is the Sony WH-1000XM6 currently available at Best Buy Canada?"

    result = await agent.run(task=task)

    print("\n--- AVAILABILITY AGENT RESPONSE ---\n")

    for message in result.messages:
        print(f"\n[{type(message).__name__}]")
        print(f"{message.source}:")
        print(message.content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())