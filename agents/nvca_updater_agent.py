import requests
from bs4 import BeautifulSoup
import logging

class NVCAUpdaterAgentV2:
    def __init__(self, nvca_url="https://nvca.org/nvca-members/"):
        self.url = nvca_url

    def fetch_vc_records(self):
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            firms = []

            for a in soup.select("a[href*='http']"):
                href = a.get("href")
                name = a.text.strip()
                if href and name:
                    firms.append({"name": name, "url": href})

            return firms[:200]  # Optional limit
        except Exception as e:
            logging.error(f"NVCA scrape failed: {e}")
            return []
