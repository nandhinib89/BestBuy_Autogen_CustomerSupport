from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.manager_agent import create_manager_agent
from agents.order_agent import create_order_agent
from agents.product_agent import create_product_agent
from agents.payment_agent import create_payment_agent
from agents.support_agent import create_support_agent
from agents.availability_agent import create_availability_agent


def create_team(model_client: OpenAIChatCompletionClient):

    # Create all agents
    manager_agent = create_manager_agent(model_client)
    order_agent = create_order_agent(model_client)
    product_agent = create_product_agent(model_client)
    payment_agent = create_payment_agent(model_client)
    support_agent = create_support_agent(model_client)
    availability_agent = create_availability_agent(model_client)

    # Terminate when ManagerAgent produces the final response
    # containing TERMINATE, or when the conversation becomes too long.
    termination = (
        TextMentionTermination("TERMINATE")
        | MaxMessageTermination(max_messages=12)
    )

    selector_prompt = """
You are selecting the next agent in a Best Buy Canada customer
support team.

Available agents:

{roles}

Participants:
{participants}

Conversation history:
{history}

Rules:

1. ManagerAgent must be the first speaker.

2. On the first turn, ManagerAgent must analyze the request and
   identify the appropriate specialist or specialists.

3. ManagerAgent should NOT provide a substantive domain answer
   before the relevant specialist has responded.

4. For a single-domain question, select the appropriate specialist
   after ManagerAgent has analyzed the request.

5. For a multi-domain question, select each relevant specialist
   one at a time.

6. Select OrderAgent for:
   - Order status
   - Order numbers
   - Order cancellation
   - Order editing
   - Shipping
   - Delivery
   - Tracking
   - Other order-related questions

7. Select ProductAgent for:
   - Product information
   - Product specifications
   - General product information
   - General product availability guidance
   - General pickup availability guidance

8. Select AvailabilityAgent for:
   - CURRENT product availability
   - CURRENT online availability
   - CURRENT shipping availability
   - CURRENT pickup availability
   - Whether a product is currently in stock
   - Whether a product is currently available for pickup

9. IMPORTANT DISTINCTION:

   ProductAgent handles general or documented information about
   how product availability or pickup works.

   AvailabilityAgent handles CURRENT product availability using
   live Best Buy Canada availability data.

   If the user asks whether a specific product is currently
   available, select AvailabilityAgent rather than ProductAgent.

10. Select PaymentAgent for:
    - Payment methods
    - Payment-related questions
    - General payment information

11. Select SupportAgent for:
    - Returns
    - Exchanges
    - Defective products
    - Marketplace returns
    - Large-item returns
    - Repairs

12. Do not select an unrelated specialist.

13. For a multi-domain question, select every relevant specialist
    needed to answer the request.

14. Select only ONE agent at a time.

15. ManagerAgent should only be selected again after the relevant
    specialist responses have been provided.

16. Do not select ManagerAgent repeatedly before a specialist has
    provided the required information.

17. After all required specialists have provided sufficient
    information, select ManagerAgent so it can synthesize the
    final customer-facing response.

18. Do not select a specialist again if that specialist has already
    provided sufficient information for the current request.

19. ManagerAgent must not invent information that was not provided
    by the relevant specialist agents.

20. For current availability questions, ManagerAgent must rely on
    AvailabilityAgent's live availability result rather than
    ProductAgent's general availability guidance.

21. If the request is ambiguous and the correct specialist cannot
    be determined, allow ManagerAgent to ask the customer for
    clarification.

22. After ManagerAgent provides a final response containing
    TERMINATE, the conversation must end.

Return only the agent name.
"""

    team = SelectorGroupChat(
        participants=[
            manager_agent,
            order_agent,
            product_agent,
            payment_agent,
            support_agent,
            availability_agent,
        ],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
    )

    return team