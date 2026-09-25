import asyncio
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from orchestration import create_team
from guardrails.input_guardrails import validate_input
from guardrails.output_guardrails import validate_output
from rag.retriever import search

from evaluations.test_cases import TEST_CASES
from evaluations.custom_checks import run_custom_checks


load_dotenv()


# =========================================================
# AutoGen chatbot
# =========================================================

async def run_chatbot(question: str):

    allowed, message = validate_input(question)

    if not allowed:
        return message

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:

        team = create_team(model_client)

        result = await team.run(task=question)

        final_message = result.messages[-1].content

        safe, validated_response = validate_output(
            final_message
        )

        return validated_response

    finally:

        await model_client.close()


def get_chatbot_result(question: str):
    """
    Run AutoGen synchronously so DeepEval can run
    outside the AutoGen event loop.
    """

    return asyncio.run(
        run_chatbot(question)
    )


# =========================================================
# Main evaluation
# =========================================================

def main():

    deepeval_cases = []

    custom_results = []

    guardrail_results = []

    # =====================================================
    # Run test cases
    # =====================================================

    for test_case_data in TEST_CASES:

        name = test_case_data["name"]
        question = test_case_data["question"]

        print("\n" + "=" * 80)
        print(f"TEST: {name}")
        print("=" * 80)

        print(f"\nQuestion:\n{question}")

        # -------------------------------------------------
        # Input guardrail test cases
        # -------------------------------------------------

        if name in {
            "out_of_scope",
            "prompt_injection",
        }:

            allowed, response = validate_input(question)

            passed = not allowed

            print(f"\nAllowed: {allowed}")
            print(f"Response:\n{response}")

            guardrail_results.append(
                {
                    "name": name,
                    "passed": passed,
                }
            )

            continue

        # -------------------------------------------------
        # Run actual chatbot
        # -------------------------------------------------

        answer = get_chatbot_result(question)

        print(f"\nAnswer:\n{answer}")

        # -------------------------------------------------
        # Custom deterministic checks
        # -------------------------------------------------

        checks = run_custom_checks(
            question,
            answer,
        )

        custom_results.append(
            {
                "name": name,
                "checks": checks,
            }
        )

        print("\nCustom checks:")

        for check in checks:

            status = (
                "PASS"
                if check["passed"]
                else "FAIL"
            )

            print(
                f"  {status:<5} "
                f"{check['name']}: "
                f"{check['reason']}"
            )

        # -------------------------------------------------
        # Retrieve evaluation context
        # -------------------------------------------------

        domain = test_case_data["domain"]

        if domain:

            retrieved_documents = search(
                question,
                k=4,
                domain=domain,
            )

        else:

            retrieved_documents = search(
                question,
                k=4,
            )

        retrieved_context = [
            document.page_content
            for document in retrieved_documents
        ]

        print(
            f"\nRetrieved chunks: "
            f"{len(retrieved_context)}"
        )

        for i, context in enumerate(
            retrieved_context,
            start=1,
        ):

            print(
                f"\n[Context {i}]"
            )

            print(
                context[:300]
                .replace("\n", " ")
            )

        # -------------------------------------------------
        # DeepEval test case
        # -------------------------------------------------

        deepeval_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=retrieved_context,
            )
        )

    # =====================================================
    # DeepEval metrics
    # =====================================================

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini",
        include_reason=True,
    )

    # =====================================================
    # Run DeepEval
    # =====================================================

    print("\n\n")
    print("=" * 80)
    print("RUNNING DEEPEVAL")
    print("=" * 80)

    evaluate(
        deepeval_cases,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
    )

    # =====================================================
    # Custom evaluation summary
    # =====================================================

    print("\n\n")
    print("=" * 80)
    print("CUSTOM EVALUATION SUMMARY")
    print("=" * 80)

    total_custom_checks = 0
    passed_custom_checks = 0

    for result in custom_results:

        for check in result["checks"]:

            total_custom_checks += 1

            if check["passed"]:
                passed_custom_checks += 1

    print(
        f"\nCustom checks passed: "
        f"{passed_custom_checks}/"
        f"{total_custom_checks}"
    )

    print("\nDetails:")

    for result in custom_results:

        print(
            f"\n{result['name']}"
        )

        for check in result["checks"]:

            status = (
                "PASS"
                if check["passed"]
                else "FAIL"
            )

            print(
                f"  {status:<5} "
                f"{check['name']}"
            )

            if not check["passed"]:

                print(
                    f"         {check['reason']}"
                )

    # =====================================================
    # Guardrail summary
    # =====================================================

    print("\n\n")
    print("=" * 80)
    print("INPUT GUARDRAIL SUMMARY")
    print("=" * 80)

    guardrails_passed = 0

    for result in guardrail_results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{status:<5} "
            f"{result['name']}"
        )

        if result["passed"]:
            guardrails_passed += 1

    print(
        f"\nGuardrail tests passed: "
        f"{guardrails_passed}/"
        f"{len(guardrail_results)}"
    )


if __name__ == "__main__":
    main()