import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

class VCWebsiteScraperAgent:
    def __init__(self, max_depth=2, max_pages=15):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.ignore_domains = [
            "linkedin.com", "twitter.com", "youtube.com",
            "facebook.com", "instagram.com"
        ]

    def _is_internal(self, link, base):
        return urlparse(link).netloc == urlparse(base).netloc

    def _is_external(self, link, base):
        return urlparse(link).netloc and urlparse(link).netloc != urlparse(base).netloc

    def _is_valid_portfolio_link(self, link):
        domain = urlparse(link).netloc.lower()
        return link.startswith("http") and not any(bad in domain for bad in self.ignore_domains)

    def scrape(self, base_url):
        visited = set()
        to_visit = [base_url]
        site_text = {}
        external_links = set()
        depth = 0

        while to_visit and len(visited) < self.max_pages and depth < self.max_depth:
            current = to_visit.pop(0)
            if current in visited or not current.startswith("http"):
                continue

            try:
                res = requests.get(current, headers=self.headers, timeout=10)
                if res.status_code != 200:
                    logging.warning(f"❌ Skipped {current} (HTTP {res.status_code})")
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                site_text[current] = text
                visited.add(current)

                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current, link['href'])
                    if self._is_internal(full_url, base_url):
                        to_visit.append(full_url)
                    elif self._is_external(full_url, base_url) and self._is_valid_portfolio_link(full_url):
                        external_links.add(full_url)

            except Exception as e:
                logging.error(f"⚠️ Error scraping {current}: {e}")
                continue

            depth += 1

        return {
            "site_text": site_text,
            "portfolio_links": list(external_links)
        }
