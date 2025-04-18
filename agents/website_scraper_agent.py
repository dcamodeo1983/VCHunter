# 🔎 VCWebsiteScraperAgentV2 – With Portfolio Link Extraction
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

class VCWebsiteScraperAgentV2:
    def __init__(self, max_depth=2, max_pages=15):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.ignore_domains = ["linkedin.com", "youtube.com", "medium.com", "twitter.com", "facebook.com", "instagram.com"]

    def _is_internal(self, link, base):
        return urlparse(link).netloc == urlparse(base).netloc

    def _is_external(self, link,
