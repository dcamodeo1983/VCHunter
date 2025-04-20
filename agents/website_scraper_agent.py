import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tldextract
import logging

class VCWebsiteScraperAgent:
    def __init__(self, keywords=None):
        self.keywords = keywords or ["portfolio", "companies", "investments", "our-companies", "team", "about"]

    def scrape(self, url):
        try:
            res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")

            links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
            portfolio_links = [link for link in links if any(k in link.lower() for k in self.keywords)]

            paragraphs = soup.find_all("p")
            text_content = {i: p.get_text(strip=True) for i, p in enumerate(paragraphs[:15])}

            return {
                "site_text": text_content,
                "portfolio_links": list(set(portfolio_links))
            }

        except Exception as e:
            logging.error(f"⚠️ Failed to scrape {url}: {e}")
            return {"site_text": {}, "portfolio_links": []}
