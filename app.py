import asyncio
import os

import streamlit as st
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

from orchestration import create_team
from guardrails.input_guardrails import validate_input
from guardrails.output_guardrails import validate_output


load_dotenv()


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Best Buy Canada Customer Support",
    page_icon="🛒",
    layout="centered",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Best Buy Canada Customer Support</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered multi-agent customer support'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Chatbot function
# ---------------------------------------------------------

async def run_chatbot(question: str):

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


def get_chatbot_response(question: str):
    """
    Run the asynchronous AutoGen workflow from Streamlit.
    """

    return asyncio.run(
        run_chatbot(question)
    )


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

prompt = st.chat_input(
    "Ask a Best Buy Canada customer-support question..."
)


if prompt:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # -----------------------------------------------------
    # Input guardrails
    # -----------------------------------------------------

    allowed, guardrail_message = validate_input(
        prompt
    )

    if not allowed:

        response = guardrail_message

    else:

        # -------------------------------------------------
        # Run AutoGen
        # -------------------------------------------------

        try:

            with st.chat_message("assistant"):

                with st.spinner(
                    "Finding the best answer..."
                ):

                    response = get_chatbot_response(
                        prompt
                    )

                st.markdown(response)

        except Exception as e:

            response = (
                "I’m sorry, but I was unable to process "
                "your request right now."
            )

            print(
                f"Application error: {e}"
            )

    # -----------------------------------------------------
    # Store assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    # -----------------------------------------------------
    # Display guardrail response if applicable
    # -----------------------------------------------------

    if not allowed:

        with st.chat_message("assistant"):
            st.markdown(response)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.subheader("Best Buy Support")

    st.write(
        "Ask questions about:"
    )

    st.markdown(
        """
        - 📦 Orders & delivery
        - 🛍️ Products
        - 💳 Payments
        - 🔄 Returns & exchanges
        - 🛠️ Repairs
        - 📍 Current availability
        """
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()