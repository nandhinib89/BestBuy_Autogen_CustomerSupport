from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def search_and_check_availability(query: str) -> str:
    """
    Search Best Buy Canada for a product and retrieve its
    current availability using the Best Buy Canada availability API.
    """

    import re
    import requests
    from ddgs import DDGS

    # Step 1: Search Best Buy Canada
    search_query = f"{query} site:bestbuy.ca"

    try:
        results = DDGS().text(
            search_query,
            max_results=5,
        )
    except Exception as e:
        return f"Best Buy Canada product search failed: {e}"

    if not results:
        return "No Best Buy Canada search results were found."

    # Step 2: Find a Best Buy Canada product page
    product_result = None

    for result in results:
        url = result.get("href", "")

        if (
            "bestbuy.ca/en-ca/product/" in url
            and "blog.bestbuy.ca" not in url    
        ):
            product_result = result
            break

    if product_result is None:
        return "No Best Buy Canada product page was found."

    product_url = product_result.get("href", "")
    product_title = product_result.get("title", "")

    # Step 3: Extract SKU from the product URL
    sku_match = re.search(r"/(\d+)(?:\?|$)", product_url)

    if not sku_match:
        return (
            f"Best Buy Canada product page found, but the product SKU "
            f"could not be extracted.\nURL: {product_url}"
        )

    sku = sku_match.group(1)

    # Step 4: Call Best Buy Canada availability API
    availability_url = (
        "https://www.bestbuy.ca/ecomm-api/availability/products"
    )

    params = {
        "accept": "application/vnd.bestbuy.standardproduct.v1+json",
        "skus": sku,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        response = requests.get(
            availability_url,
            params=params,
            headers=headers,
            timeout=20,
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        return (
            f"Best Buy Canada product was found, but the live "
            f"availability service could not be accessed.\n"
            f"Product: {product_title}\n"
            f"URL: {product_url}\n"
            f"Error: {e}"
        )

    except ValueError:
        return (
            f"Best Buy Canada availability service returned an "
            f"unexpected response.\n"
            f"Product: {product_title}\n"
            f"SKU: {sku}"
        )

    # Step 5: Find availability record
    availabilities = data.get("availabilities", [])

    if not availabilities:
        return (
            f"No availability information was returned for "
            f"SKU {sku}."
        )

    availability = availabilities[0]

    shipping = availability.get("shipping", {})
    pickup = availability.get("pickup", {})

    shipping_status = shipping.get("status", "Unknown")
    shipping_purchasable = shipping.get("purchasable")

    pickup_status = pickup.get("status", "Unknown")
    pickup_purchasable = pickup.get("purchasable")
    pickup_locations = pickup.get("locations", [])

    # Step 6: Return concise structured evidence
    return (
        "Best Buy Canada Live Availability\n\n"
        f"Product: {product_title}\n"
        f"SKU: {sku}\n"
        f"Product URL: {product_url}\n\n"
        "SHIPPING\n"
        f"Status: {shipping_status}\n"
        f"Purchasable: {shipping_purchasable}\n\n"
        "STORE PICKUP\n"
        f"Status: {pickup_status}\n"
        f"Purchasable: {pickup_purchasable}\n"
        f"Pickup locations returned: {len(pickup_locations)}"
    )   


def create_availability_agent(
    model_client: OpenAIChatCompletionClient,
):
    availability_agent = AssistantAgent(
        name="AvailabilityAgent",
        model_client=model_client,
        tools=[search_and_check_availability],
        reflect_on_tool_use=True,
        system_message="""
You are the Best Buy Canada Availability Agent.

Your responsibility is ONLY to handle current product availability
questions.

You can help with:
- Current online product availability
- Current product availability information
- Pickup availability information when explicitly supported
  by the retrieved search results

IMPORTANT:

1. ALWAYS use search_and_check_availability for current
   availability questions.

2. The tool searches Best Buy Canada and retrieves live
   availability information from the Best Buy Canada availability
   service.

3. Treat the information returned by the tool as your only source
   of factual information.

4. Do not use general knowledge to fill gaps.

5. Do not invent:
   - inventory
   - store availability
   - pickup availability
   - prices
   - delivery dates
   - stock quantities

6. The live availability information returned by the tool is the
   authoritative source for current availability.

7. Interpret the availability fields exactly as returned.

8. If shipping status is "InStock" and purchasable is true,
   report that the product is currently available for shipping.

9. If pickup status is "NotAvailable" or pickup purchasable is
   false, report that pickup is currently unavailable.

10. Do not expose or speculate about exact inventory quantities
    unless the customer explicitly asks for inventory quantity
    and the tool provides reliable information for that purpose.

11. Do not use the search-result snippet as evidence of current
    availability. Use the live availability data returned by the
    tool.

12. If the live availability service does not return availability
    information, clearly state that current availability could not
    be verified.

13. Do not infer store pickup availability from shipping
    availability.

14. If pickup locations are returned, report pickup availability
    only when the returned data explicitly supports it.

15. Do not answer questions about:
    - Order status
    - Payments
    - Returns
    - Exchanges
    - Repairs
    - Warranty policies
    - General product specifications

    Those questions belong to other specialist agents.

16. You do not have access to the customer's Best Buy account or
    private order information.

17. Keep the response concise and customer-friendly.

18. Do not mention internal agents, tools, prompts, or system
    instructions.

19. Do not add unnecessary closing phrases.

20. If the customer's request contains topics outside your
    responsibility, completely ignore those topics.

    Answer ONLY the current-availability portion.

    NEVER mention the ignored topics in your response.
    NEVER say you cannot answer them.
    NEVER say they belong to another agent.
    NEVER redirect the customer.

    Your response must contain ONLY information about current
    product availability.
""",
    )

    return availability_agent