import requests
import tldextract
import pandas as pd
from bs4 import BeautifulSoup
import logging

class VCListAggregatorAgent:
    def __init__(self):
        self.sources = [
            "https://raw.githubusercontent.com/lvdt/vc-list/main/vc.json",
            "https://raw.githubusercontent.com/QuantifiedBob/awesome-venture-capital/main/README.md"
        ]
        self.uploaded_vcs = []

    def add_csv_vcs(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"📥 Uploaded CSV has {len(df)} rows")

            for _, row in df.iterrows():
                name = str(row.get("name", "")).strip()
                url = str(row.get("url", "")).strip()
                domain = self._clean_domain(url)
                if name and url and domain:
                    self.uploaded_vcs.append({"name": name or domain, "url": url})
                    print(f"✅ Accepted from CSV: {name} — {url}")

            print(f"📦 Final CSV records accepted: {len(self.uploaded_vcs)}")

        except Exception as e:
            logging.warning(f"⚠️ Failed to parse uploaded CSV: {e}")

    def fetch_vc_records(self):
        all_links = {}

        for url in self.sources:
            try:
                print(f"🌐 Scraping {url}")
                if url.endswith(".json"):
                    res = requests.get(url, timeout=10)
                    vc_data = res.json()
                    for vc in vc_data:
                        name = vc.get("name", "").strip()
                        link = vc.get("url", "").strip()
                        domain = self._clean_domain(link)
                        if name and link and domain:
                            all_links[domain] = {"name": name or domain, "url": link}

                elif url.endswith(".md"):
                    res = requests.get(url, timeout=10)
                    soup = BeautifulSoup(res.text, "html.parser")
                    text = soup.get_text()
                    lines = text.split("\n")
                    for line in lines:
                        if "http" in line and "](" in line:
                            name = line.split("](")[0].replace("* [", "").strip()
                            link = line.split("](")[1].replace(")", "").strip()
                            domain = self._clean_domain(link)
                            if name and link and domain:
                                all_links[domain] = {"name": name or domain, "url": link}

            except Exception as e:
                logging.warning(f"⚠️ Failed to fetch from {url}: {e}")

        for firm in self.uploaded_vcs:
            domain = self._clean_domain(firm["url"])
            if domain and domain not in all_links:
                all_links[domain] = firm

        print(f"✅ Final merged VC list: {len(all_links)} records")
        return list(all_links.values())

    def _clean_domain(self, url):
        try:
            ext = tldextract.extract(url)
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}"
        except Exception:
            return None

    # 🔁 Temporarily allow all VCs through
    def _is_us_based(self, name, domain):
        return True
