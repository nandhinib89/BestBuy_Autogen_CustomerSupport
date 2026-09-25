import requests

SKU = "19320384"

url = "https://www.bestbuy.ca/ecomm-api/availability/products"

params = {
    "accept": "application/vnd.bestbuy.standardproduct.v1+json",
    "skus": SKU,
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

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=20,
)

print("Status:", response.status_code)
print("URL:", response.url)
print("\nResponse:")
print(response.text)