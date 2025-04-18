import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

class VCWebsiteScraperAgent:
    def __init__(self, max_depth=2, max_pages=15):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.ignore_domains = ["linkedin.com", "twitter.com", "youtube.com", "facebook.com", "instagram.com"]

    def _is_internal(self, link, base):
        return urlparse(link).netloc == urlparse(base).netloc

    def _is_external(self, link, base):
        return urlparse(link).netloc and urlparse(link).netloc != urlparse(base).netloc

    def _is_valid_portfolio_link(self, link):
        domain = urlparse(link).netloc.lower()
        return link.startswith("http") and not any(bad in domain for bad in self.ignore_domains)

    def scrape(self, base_url_
