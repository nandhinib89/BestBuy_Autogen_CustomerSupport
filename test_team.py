import asyncio
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from orchestration import create_team
from guardrails.input_guardrails import validate_input
from guardrails.output_guardrails import validate_output


load_dotenv()


async def main():

    task = "Is the Sony WH-1000XM6 currently available?"

    # Input guardrails
    allowed, message = validate_input(task)

    if not allowed:
        print("\n--- GUARDRAIL RESPONSE ---\n")
        print(message)
        return

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    team = create_team(model_client)

    result = await team.run(task=task)

    # Get the final message from the team
    final_message = result.messages[-1].content

    # Remove AutoGen termination marker from customer-facing output
    final_message = final_message.replace("TERMINATE", "").strip()

    # Output guardrail
    safe, validated_response = validate_output(final_message)

    print("\n--- FINAL RESPONSE ---\n")
    print(validated_response)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())