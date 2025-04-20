# 📦 VCListAggregatorAgent – Provides VC URLs (from CSV or fallback list)
import logging

class VCListAggregatorAgent:
    def __init__(self):
        self.static_vcs = [
            {"name": "Andreessen Horowitz", "url": "https://a16z.com"},
            {"name": "Benchmark Capital", "url": "https://www.benchmark.com"},
            {"name": "Sequoia Capital", "url": "https://www.sequoiacap.com"},
            {"name": "First Round Capital", "url": "https://firstround.com"},
            {"name": "Greylock Partners", "url": "https://greylock.com"},
            {"name": "Union Square Ventures", "url": "https://www.usv.com"},
            {"name": "Lightspeed Venture Partners", "url": "https://lsvp.com"},
            {"name": "Bessemer Venture Partners", "url": "https://www.bvp.com"},
            {"name": "Accel", "url": "https://www.accel.com"},
            {"name": "Foundry Group", "url": "https://www.foundrygroup.com"},
            {"name": "General Catalyst", "url": "https://www.generalcatalyst.com"},
            {"name": "Initialized Capital", "url": "https://initialized.com"},
            {"name": "Menlo Ventures", "url": "https://www.menlovc.com"},
            {"name": "NEA (New Enterprise Associates)", "url": "https://www.nea.com"},
            {"name": "Upfront Ventures", "url": "https://upfront.com"},
            {"name": "Homebrew", "url": "https://homebrew.co"},
            {"name": "Pear VC", "url": "https://pear.vc"},
            {"name": "Susa Ventures", "url": "https://www.susaventures.com"},
            {"name": "Kleiner Perkins", "url": "https://www.kleinerperkins.com"},
            {"name": "Craft Ventures", "url": "https://www.craftventures.com"},
            {"name": "Lux Capital", "url": "https://www.luxcapital.com"},
            {"name": "Tribe Capital", "url": "https://www.tribecap.co"},
            {"name": "SignalFire", "url": "https://www.signalfire.com"},
            {"name": "Village Global", "url": "https://www.villageglobal.vc"},
            {"name": "DCVC", "url": "https://www.dcvc.com"},
            {"name": "Costanoa Ventures", "url": "https://www.costanoavc.com"},
            {"name": "The House Fund", "url": "https://www.thehouse.fund"},
            {"name": "Eclipse Ventures", "url": "https://eclipse.vc"},
            {"name": "Canvas Ventures", "url": "https://canvas.vc"},
            {"name": "Defy.vc", "url": "https://defy.vc"}
        ]
        self.csv_vcs = []

    def add_csv_vcs(self, csv_path):
        logging.warning("⚠️ Skipping CSV parsing (using hardcoded VC list only).")
        return

    def fetch_vc_records(self):
        logging.info(f"✅ Returning {len(self.static_vcs)} static VC records")
        return self.static_vcs
