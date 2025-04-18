# 🔄 NVCAUpdaterAgentV2 – Full VC Record Extractor
import requests
from bs4 import BeautifulSoup
import logging

class NVCAUpdaterAgentV2:
    def __init__(self, base_url="https://nvca.org/nvca-members/"):
        self.base_url = base_url
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def fetch_vc_records(self):
        try:
            res = requests.get(self.base_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            vc_blocks = soup.find_all("div", class_="member-list__item")
            records = []

            for block in vc_blocks:
                name_tag = block.find("h3")
                link_tag = block.find("a", href=True)
                loc_tag = block.find("div", class_="member-list__location")

                vc_name = name_tag.text.strip() if name_tag else "Unknown VC"
                vc_url = link_tag['href'].strip() if link_tag else None
                vc_location = loc_tag.text.strip() if loc_tag else None

                if vc_url and vc_url.startswith("http"):
                    records.append({
                        "name": vc_name,
                        "url": vc_url,
                        "location": vc_location
                    })

            return records
        except Exception as e:
            logging.error(f"Failed to fetch NVCA members: {e}")
            return []

