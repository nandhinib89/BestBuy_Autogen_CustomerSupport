import re


INTERNAL_CONTENT_PATTERNS = [
    r"\bsystem prompt\b",
    r"\bsystem instructions\b",
    r"\binternal instructions\b",
    r"\binternal tools\b",
    r"\btool call\b",
    r"\btool calls\b",
    r"\bfunction call\b",
    r"\bfunction calls\b",
]


def validate_output(response: str) -> tuple[bool, str]:
    """
    Validate and clean the final customer-facing response.
    """

    if not response or not response.strip():
        return False, "I’m unable to provide a response at this time."

    # Remove AutoGen termination marker
    response = response.replace("TERMINATE", "").strip()

    # Remove a standalone period accidentally appended by the agent
    response = re.sub(r"\s*\n\s*\.\s*$", "", response).strip()

    normalized_response = response.lower()

    for pattern in INTERNAL_CONTENT_PATTERNS:
        if re.search(pattern, normalized_response):
            return (
                False,
                "I’m unable to provide that information.",
            )

    return True, response