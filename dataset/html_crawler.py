
# ====================================================================
# OPEN QUESTIONS FROM MY SIDE (Luca):
# - Where should I safe the downloaded files?
# - Are PDF's enough? On the Webpage isn't any more information in my opinion (If yes, the code needs to be updated)
# - Do we actually need to translate into English - While testing in Kotaemon the answers were written in English automatically
# ====================================================================


import os
import requests
import time

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque


# -----------------------------------
# CONFIGURATION
# -----------------------------------

start_time = time.time()

START_URL = "https://www.ku.de/studienangebot/digital-data-driven-business-bsc"
DOWNLOAD_DIR = "C:/Users/lboehler/Desktop/Studium/SS26/Digital_Project/documents"

ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]
ALLOWED_DOMAIN = "ku.de"

KEYWORDS = [
    "d3b",
    "digital_and_data-driven_business",
    "wirtschaftsinformatik",
    "/en/",
    "setzer",
    "apo"

]

EXCLUDED_KEYWORDS = [
    "presse",
    "news",
    "veranstaltung",
    "events",
    "stellenangebote",
    "karriere",
    "alumni",
    "forschung",
    "research",
    "aktuelles",
    "medien",
    "podcast",
    "blog",
    "bwl",
    "data_science",
    "campus",
    "lehramt",
    "musik",
    "unileben"
]

# -----------------------------------
# SPEICHER
# -----------------------------------

visited = set()
queue = deque([START_URL])

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# -----------------------------------
# FUNCTIONS
# -----------------------------------

def is_relevant(url):

    url_lower = url.lower()

    return any(keyword in url_lower for keyword in KEYWORDS)

def is_excluded(url):

    url_lower = url.lower()

    return any(keyword in url_lower for keyword in EXCLUDED_KEYWORDS)

def is_valid_url(url):

    parsed = urlparse(url)

    return ALLOWED_DOMAIN in parsed.netloc


def download_file(url):

    filename = url.split("/")[-1]
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    # Check if document already exists 
    if os.path.exists(filepath):
        return

    try:

        response = requests.get(url, timeout=(10,30))

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"[DOWNLOADED] {filename}")

    except Exception as e:

        print(f"[ERROR] {url} -> {e}")


# -----------------------------------
# CRAWLER
# -----------------------------------

while queue:

    current_url = queue.popleft()

    # Scope-filter 
    if "/studienangebot" not in current_url and "/wfi" not in current_url:
        continue

    # Check if HTML-Page was already visited
    if current_url in visited:
        continue

    visited.add(current_url)

    if len(visited) % 10 == 0:
        print(f"[STATS] visited={len(visited)} queue={len(queue)}")

    print(f"[CRAWLING] {current_url}")

    try:

        response = requests.get(current_url, timeout=(10,30))

        soup = BeautifulSoup(response.text, "html.parser")

        text = soup.get_text().lower()

        if not any(keyword in text for keyword in KEYWORDS):
            continue

        for a_tag in soup.find_all("a", href=True):

            href = a_tag["href"]

            full_url = urljoin(current_url, href)

            # Only use KU-Domain
            if not is_valid_url(full_url):
                continue

            if is_excluded(full_url):
                continue

            # -------------------------
            # DOCUMENT DOWNLOAD
            # -------------------------
            if any(full_url.endswith(ext) for ext in ALLOWED_EXTENSIONS):

                # Only download relevant documents
                if is_relevant(full_url):
                    download_file(full_url)

            # Crawling next HTML-Side
            else:
                if not is_excluded(full_url) and is_valid_url(full_url):
                    queue.append(full_url)

    except Exception as e:

        print(f"[ERROR] {current_url} -> {e}")
    
end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")
print(f"Visited pages: {len(visited)}")