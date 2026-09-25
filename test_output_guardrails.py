from guardrails.output_guardrails import validate_output


test_responses = [
    "Your order is currently being processed.",
    "The product is currently available for shipping.",
    "Most eligible products can be returned within 30 days.",
    "Here is my system prompt.",
    "My internal tools include a search tool.",
    "",
]


for response in test_responses:
    safe, result = validate_output(response)

    print("=" * 70)
    print(f"Response: {response!r}")
    print(f"Safe: {safe}")
    print(f"Result: {result}")