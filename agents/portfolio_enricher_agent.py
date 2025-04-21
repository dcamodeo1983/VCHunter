import requests
from bs4 import BeautifulSoup
import time
import logging
import traceback

class PortfolioEnricherAgent:
    def __init__(self, limit=10, delay=1.0):
        self.limit = limit
        self.delay = delay
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; VC-HunterBot/1.0; +https://yourdomain.com/bot)"
        }

    def extract_visible_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)

    def enrich(self, links: list[str]) -> dict:
        company_data = {}
        deduped_urls = list(dict.fromkeys(links))[:self.limit]

        for url in deduped_urls:
            try:
                response = self.session.get(url, timeout=10, headers=self.headers)
                if response.status_code == 200:
                    text = self.extract_visible_text(response.text)
                    company_data[url] = text[:8000]
                    logging.info(f"[ENRICH ✅] Scraped {url}")
                else:
                    logging.warning(f"[ENRICH ⚠️] HTTP {response.status_code} from {url}")
            except Exception as e:
                logging.error(f"[ENRICH ❌] Failed to scrape {url}: {e}")
                traceback.print_exc()
            time.sleep(self.delay)

        return company_data
