class VCListAggregatorAgent:
    def __init__(self):
        self.static_vcs = [
            {"name": "USV", "url": "https://www.usv.com"},
            {"name": "Andreessen Horowitz", "url": "https://a16z.com"},
            {"name": "First Round", "url": "https://firstround.com"}
        ]

    def add_csv_vcs(self, csv_path):
        # This version ignores uploaded CSVs completely
        print("⚠️ Skipping CSV parsing (hardcoded VC list used)")

    def fetch_vc_records(self):
        print(f"✅ Returning {len(self.static_vcs)} static VC records")
        return self.static_vcs
