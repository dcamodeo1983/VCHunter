# 🏢 PortfolioEnricherAgentV3 – Fully Automated from Detected Links
import requests
from bs4 import BeautifulSoup
import time

class PortfolioEnricherAgentV3:
    def __init__(self, delay=1.0):
        self.session = requests.Session()
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.delay = delay

    def _extract_visible_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return " ".join(soup.stripped_strings)

    def enrich(self, portfolio_urls):
        data = {}
        for url in set(portfolio_urls):
            try:
                res = self.session.get(url, headers=self.headers, timeout=10)
                if res.status_code == 200:
                    text = self._extract_visible_text(res.text)[:8000]
                    data[url] = text
                    print(f"✅ {url} – {len(text)} chars")
                else:
                    print(f"⚠️ {url} returned {res.status_code}")
            except Exception as e:
                print(f"❌ Failed to scrape {url}: {e}")
            time.sleep(self.delay)
        return data
# Real content from PortfolioEnricherAgentV3
