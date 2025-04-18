import requests
from bs4 import BeautifulSoup
import time
import logging

class PortfolioEnricherAgent:
    def __init__(self, limit=10):
        self.limit = limit
        self.session = requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def extract_visible_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)

    def enrich(self, links):
        company_data = {}
        for url in list(set(links))[:self.limit]:
            try:
                r = self.session.get(url, timeout=10, headers=self.headers)
                if r.status_code == 200:
                    text = self.extract_visible_text(r.text)
                    company_data[url] = text[:8000]  # Clip to manageable size
                    logging.info(f"✅ Scraped {url}")
                else:
                    logging.warning(f"⚠️ {url} returned HTTP {r.status_code}")
            except Exception as e:
                logging.error(f"❌ Error scraping {url}: {e}")
            time.sleep(1)  # Rate limiting for polite scraping
        return company_data
