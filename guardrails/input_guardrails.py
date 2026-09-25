import re


BLOCKED_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"ignore (your|the) system prompt",
    r"reveal (your|the) system prompt",
    r"show (your|the) system prompt",
    r"tell me (your|the) system prompt",
    r"reveal (your|the) internal instructions",
    r"show me (your|the) internal instructions",
    r"what are your internal tools",
    r"list your tools",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b",
    r"\bstock market\b",
    r"\bpolitics\b",
    r"\brecipe\b",
    r"\bfootball\b",
    r"\bsoccer\b",
]


def check_scope(user_query: str) -> tuple[bool, str]:
    """
    Check whether a query is clearly outside Best Buy
    customer-support scope.

    Returns:
        (is_in_scope, message)
    """

    normalized_query = user_query.lower().strip()

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, normalized_query):
            return (
                False,
                "I can help with Best Buy Canada customer-support questions "
                "such as orders, products, payments, returns, repairs, and "
                "availability.",
            )

    return True, ""


def check_input(user_query: str) -> tuple[bool, str]:
    """
    Validate a user query before sending it to the agent team.

    Returns:
        (is_allowed, message)
    """

    if not user_query or not user_query.strip():
        return False, "Please enter a question."

    normalized_query = user_query.lower().strip()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, normalized_query):
            return (
                False,
                "I can help with Best Buy Canada customer-support questions, "
                "but I can't provide internal instructions or system details.",
            )

    return True, ""

def validate_input(user_query: str) -> tuple[bool, str]:
    """
    Run all input guardrails.
    """

    allowed, message = check_input(user_query)

    if not allowed:
        return False, message

    in_scope, message = check_scope(user_query)

    if not in_scope:
        return False, message

    return True, ""