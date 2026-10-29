from dotenv import load_dotenv

load_dotenv()

import requests
import trafilatura


url = "https://m.cricbuzz.com/live-cricket-scores/151532/ind-vs-wi-1st-odi-west-indies-tour-of-india-2026"

# Fetch webpage HTML
response = requests.get(
    url,
    headers={"User-Agent": "WebRAG/1.0"},
    timeout=15
)

response.raise_for_status()

html = response.text

print("HTML characters:", len(html))

# Extract main textual content
text = trafilatura.extract(
    html,
    include_links=False,
    include_images=False,
    include_tables=True
)

if not text:
    raise ValueError("Could not extract webpage content")

print("Extracted characters:", len(text))

print("\nFirst 3000 characters:")
print(text[:3000])