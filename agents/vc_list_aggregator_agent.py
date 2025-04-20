# vc_list_aggregator_agent.py — Aggregates VC list from web or returns hardcoded fallback

class VCListAggregatorAgent:
    def __init__(self):
        self.hardcoded_list = [
            {"name": "Sequoia Capital", "url": "https://www.sequoiacap.com"},
            {"name": "Andreessen Horowitz", "url": "https://a16z.com"},
            {"name": "Benchmark", "url": "https://benchmark.com"},
            {"name": "Accel", "url": "https://www.accel.com"},
            {"name": "Index Ventures", "url": "https://www.indexventures.com"},
            {"name": "Bessemer Venture Partners", "url": "https://www.bvp.com"},
            {"name": "Lightspeed Venture Partners", "url": "https://lsvp.com"},
            {"name": "Greylock Partners", "url": "https://greylock.com"},
            {"name": "Kleiner Perkins", "url": "https://www.kleinerperkins.com"},
            {"name": "General Catalyst", "url": "https://www.generalcatalyst.com"},
            {"name": "Union Square Ventures", "url": "https://www.usv.com"},
            {"name": "First Round Capital", "url": "https://firstround.com"},
            {"name": "NEA", "url": "https://www.nea.com"},
            {"name": "GV (Google Ventures)", "url": "https://www.gv.com"},
            {"name": "CRV", "url": "https://www.crv.com"},
            {"name": "Battery Ventures", "url": "https://www.battery.com"},
            {"name": "DFJ Growth", "url": "https://dfj.com"},
            {"name": "IVP", "url": "https://www.ivp.com"},
            {"name": "Forerunner Ventures", "url": "https://www.forerunnerventures.com"},
            {"name": "Menlo Ventures", "url": "https://www.menlovc.com"},
            {"name": "Wing Venture Capital", "url": "https://www.wing.vc"},
            {"name": "SignalFire", "url": "https://www.signalfire.com"},
            {"name": "Uncork Capital", "url": "https://www.uncorkcapital.com"},
            {"name": "Lux Capital", "url": "https://www.luxcapital.com"},
            {"name": "Slow Ventures", "url": "https://slow.co"},
            {"name": "Initialized Capital", "url": "https://initialized.com"},
            {"name": "BoxGroup", "url": "https://www.boxgroup.com"},
            {"name": "Homebrew", "url": "https://homebrew.co"},
            {"name": "Upfront Ventures", "url": "https://upfront.com"},
            {"name": "DCVC", "url": "https://www.dcvc.com"},
        ]

    def fetch_vc_records(self):
        # You can later add GitHub/Dealroom logic here if `trigger_nvca` is True
        return self.hardcoded_list
