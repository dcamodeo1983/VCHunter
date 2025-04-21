import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import logging

class VCWebsiteScraperAgent:
    def __init__(self, keywords=None, max_paragraphs=15):
        self.keywords = keywords or ["portfolio", "companies", "investments", "our-companies", "team", "about"]
        self.max_paragraphs = max_paragraphs
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def scrape(self, url):
        """
        Scrapes a VC firm's homepage for:
            - Portfolio-related links
            - Main visible paragraph text for basic summarization

        Args:
            url (str): Root VC firm website

        Returns:
            dict: {
                "site_text": dict of paragraph text,
                "portfolio_links": list of URLs
            }
        """
        try:
            res = self.session.get(url, timeout=10)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, "html.parser")

            # Extract and normalize all links
            all_links = [
                urljoin(url, a["href"])
                for a in soup.find_all("a", href=True)
                if a["href"].startswith("/") or urlparse(a["href"]).netloc
            ]
            all_links = list(set(all_links))  # Deduplicate

            # Filter links based on portfolio-related keywords
            portfolio_links = [
                link for link in all_links
                if any(k in link.lower() for k in self.keywords)
            ]

            # Extract and clean visible paragraph
