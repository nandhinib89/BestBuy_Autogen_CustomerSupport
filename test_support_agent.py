import asyncio
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.support_agent import create_support_agent


load_dotenv()


async def main():

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_support_agent(model_client)

    question = "Can I return a Marketplace product to a Best Buy store?"

    result = await agent.run(task=question)

    print("\n--- SUPPORT AGENT RESPONSE ---\n")

    print(result.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())