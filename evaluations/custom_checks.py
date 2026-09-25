import re


def check_no_unsupported_eligibility_claim(
    question: str,
    answer: str,
) -> tuple[bool, str]:
    """
    Detect potentially unsupported product-specific eligibility claims.

    This is a deterministic application-specific check.
    It does not determine the actual Best Buy policy.
    """

    question_lower = question.lower()
    answer_lower = answer.lower()

    # Product-specific indicators.
    product_indicators = [
        "sony",
        "headphones",
        "laptop",
        "tv",
        "television",
        "phone",
        "iphone",
        "macbook",
        "wh-1000xm6",
    ]

    support_terms = [
        "return",
        "exchange",
        "repair",
        "warranty",
        "eligible",
        "qualify",
    ]

    is_product_specific = any(
        term in question_lower
        for term in product_indicators
    )

    asks_support_question = any(
        term in question_lower
        for term in support_terms
    )

    if not (is_product_specific and asks_support_question):
        return True, "Not applicable."

    # ---------------------------------------------------------
    # Strong product-specific eligibility claims
    # ---------------------------------------------------------

    unsupported_claim_patterns = [
        r"\byou can return\b",
        r"\byou can exchange\b",
        r"\byou can repair\b",
        r"\bis eligible for\b",
        r"\bqualifies for\b",
        r"\bis covered by\b",
        r"\bis covered under\b",
        r"\bcan be returned\b",
        r"\bcan be exchanged\b",
        r"\bcan be repaired\b",
    ]

    for pattern in unsupported_claim_patterns:
        if re.search(pattern, answer_lower):
            return (
                False,
                "Potentially unsupported product-specific eligibility claim.",
            )

    # ---------------------------------------------------------
    # Product-specific return-window claims
    # ---------------------------------------------------------

    return_window_claim = re.search(
        r"\b(return|returned|returning)\b.{0,80}"
        r"\b(30 days|30-day|thirty days)\b",
        answer_lower,
    )

    if return_window_claim:
        return (
            False,
            "Response appears to apply a general return window "
            "to a specific product.",
        )

    return True, "No obvious unsupported eligibility claim detected."


def check_no_account_access_claim(
    answer: str,
) -> tuple[bool, str]:
    """
    Ensure the chatbot does not claim access to private
    customer information.
    """

    answer_lower = answer.lower()

    patterns = [
        r"\bi checked your order\b",
        r"\bi checked your account\b",
        r"\bi verified your order\b",
        r"\bi verified your account\b",
        r"\bi can see your order\b",
        r"\bi can see your account\b",
        r"\byour refund has been processed\b",
        r"\byour payment has been refunded\b",
        r"\byour order is currently\b",
    ]

    for pattern in patterns:
        if re.search(pattern, answer_lower):
            return (
                False,
                "Response appears to claim access to private customer information.",
            )

    return True, "No private-account access claim detected."


def check_no_internal_information(
    answer: str,
) -> tuple[bool, str]:
    """
    Ensure customer-facing output does not expose internal
    implementation details.
    """

    answer_lower = answer.lower()

    patterns = [
        "system prompt",
        "system instructions",
        "internal instructions",
        "internal tools",
        "tool call",
        "function call",
        "autogen",
        "selector group chat",
        "faiss",
        "cross-encoder",
    ]

    for pattern in patterns:
        if pattern in answer_lower:
            return (
                False,
                f"Internal implementation detail detected: '{pattern}'.",
            )

    return True, "No internal implementation details detected."


def run_custom_checks(
    question: str,
    answer: str,
) -> list[dict]:
    """
    Run all deterministic application-specific checks.
    """

    checks = [
        (
            "unsupported_eligibility",
            *check_no_unsupported_eligibility_claim(
                question,
                answer,
            ),
        ),
        (
            "account_access",
            *check_no_account_access_claim(answer),
        ),
        (
            "internal_information",
            *check_no_internal_information(answer),
        ),
    ]

    return [
        {
            "name": name,
            "passed": passed,
            "reason": reason,
        }
        for name, passed, reason in checks
    ]