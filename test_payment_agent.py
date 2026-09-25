import asyncio
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.payment_agent import create_payment_agent


load_dotenv()


async def main():

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_payment_agent(model_client)

    question = "My order was cancelled but I was still charged. What happened?"

    result = await agent.run(task=question)

    print("\n--- PAYMENT AGENT RESPONSE ---\n")

    print(result.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())