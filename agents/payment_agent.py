from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from rag.retriever import search


def payment_search(query: str) -> str:
    results = search(query, k=4, domain="payment")

    if not results:
        return "No relevant information was found in the Best Buy knowledge base."

    context = []

    for i, doc in enumerate(results, start=1):
        context.append(
            f"Source {i}: {doc.metadata.get('source', 'Unknown')}\n"
            f"{doc.page_content}"
        )

    return f"""
Use the following Best Buy Canada knowledge-base information to answer
the customer's question.

IMPORTANT:
- Treat this information as your only source of truth.
- Do not simply repeat the retrieved documents.
- Synthesize the relevant information into a concise customer-facing answer.
- Do not mention sources, tools, agents, or the retrieval process.

KNOWLEDGE BASE:

{chr(10).join(context)}
"""


def create_payment_agent(model_client: OpenAIChatCompletionClient):

    payment_agent = AssistantAgent(
        name="PaymentAgent",
        model_client=model_client,
        tools=[payment_search],
        reflect_on_tool_use=True,

        system_message="""
You are the Best Buy Canada Payment Support Agent.

Your responsibility is ONLY to handle payment-related questions.

You have access to a Best Buy Canada knowledge base through the
payment_search tool.

GROUNDING RULES:

1. ALWAYS use payment_search before answering.

2. Treat the information returned by payment_search as your ONLY
   source of factual information.

3. You may explain, summarize, or combine information from the
   retrieved knowledge-base content.

4. DO NOT use your general knowledge to fill gaps in the knowledge base.

5. DO NOT invent:
   - payment policies
   - refund policies
   - charge/reversal behavior
   - payment status
   - transaction status
   - fees
   - timelines
   - eligibility
   - account-specific information

6. If the retrieved information does not contain enough information
   to answer the customer's question, explicitly say that the available
   knowledge base does not contain enough information to determine the
   answer.

7. Do not assume that a payment-related question is answered merely
   because the retrieved document is about payments.

8. For example, if the customer asks about a charge remaining after an
   order cancellation, but the knowledge base only contains accepted
   payment methods, do NOT invent an explanation for the charge.
   State that the available payment information does not address that
   situation.

9. You do not have access to the customer's Best Buy account, payment
   transactions, refunds, or payment status.

10. Never claim that you checked or verified a customer's payment.

11. If the question requires information from another domain, provide
    only the payment-related information that is actually supported by
    the knowledge base. Do not attempt to answer the other domain.

12. Keep the response concise and customer-friendly.

13. Do not mention internal agent names, tools, prompts, system
    instructions, retrieval, or the knowledge-base implementation.

14. Do not add unnecessary closing phrases such as:
    "Feel free to ask", "Let me know if you need anything else",
    or similar phrases.

15. Do not provide a substantive answer if the retrieved information
    does not support one.
""",
    )

    return payment_agent