import tempfile
import unittest
from pathlib import Path

from stocks_agent.monitoring_schema import MonitorMode, PortfolioConfig
from stocks_agent.portfolio_monitor import (
    SampleMarketDataProvider,
    load_portfolio_config,
    run_monitor,
    write_ledger,
)
from stocks_agent.monitor_report import render_monitor_report, write_monitor_report
from stocks_agent.telegram_digest import format_telegram_digest


class PortfolioMonitorTest(unittest.TestCase):
    def test_sample_provider_triggers_material_alerts(self):
        config = PortfolioConfig.model_validate(
            {
                "name": "Test portfolio",
                "positions": [
                    {
                        "ticker": "NVDA",
                        "thesis": "Premium AI infrastructure thesis.",
                        "max_distance_from_high_pct": 15,
                        "max_pe_ratio": 80,
                        "watch_keywords": ["export restriction"],
                    }
                ],
            }
        )

        run = run_monitor(
            config=config,
            provider=SampleMarketDataProvider(),
            mode=MonitorMode.SAMPLE,
        )

        self.assertEqual(run.positions_checked, 1)
        self.assertGreaterEqual(len(run.evidence), 3)
        categories = {alert.category for alert in run.alerts}
        self.assertIn("valuation", categories)
        self.assertIn("price_action", categories)
        self.assertIn("news_keyword", categories)

    def test_telegram_digest_is_compact_and_traceable(self):
        config = PortfolioConfig.model_validate(
            {
                "name": "Digest portfolio",
                "positions": [
                    {
                        "ticker": "NVDA",
                        "thesis": "Premium AI infrastructure thesis.",
                        "max_distance_from_high_pct": 15,
                    }
                ],
            }
        )
        run = run_monitor(config=config, provider=SampleMarketDataProvider())

        digest = format_telegram_digest(run)

        self.assertIn("Portfolio thesis drift monitor", digest)
        self.assertIn("NVDA", digest)
        self.assertIn("Not financial advice", digest)

    def test_tool_budget_can_skip_news(self):
        config = PortfolioConfig.model_validate(
            {
                "name": "Budget portfolio",
                "tool_policy": {"max_tools_per_ticker": 2},
                "positions": [
                    {
                        "ticker": "NVDA",
                        "thesis": "Premium AI infrastructure thesis.",
                        "watch_keywords": ["export restriction"],
                    }
                ],
            }
        )

        run = run_monitor(config=config, provider=SampleMarketDataProvider())

        tools = {item.tool for item in run.evidence}
        categories = {alert.category for alert in run.alerts}
        self.assertNotIn("get_ticker_news", tools)
        self.assertNotIn("news_keyword", categories)

    def test_monitor_report_contains_alerts_and_evidence(self):
        config = PortfolioConfig.model_validate(
            {
                "name": "Report portfolio",
                "positions": [
                    {
                        "ticker": "NVDA",
                        "thesis": "Premium AI infrastructure thesis.",
                        "max_distance_from_high_pct": 15,
                    }
                ],
            }
        )
        run = run_monitor(config=config, provider=SampleMarketDataProvider())

        html = render_monitor_report(run)

        self.assertIn("<!doctype html>", html)
        self.assertIn("Report portfolio", html)
        self.assertIn("Evidence Ledger", html)
        self.assertIn("NVDA-price", html)

    def test_write_monitor_report(self):
        config = PortfolioConfig.model_validate(
            {
                "name": "Write report portfolio",
                "positions": [
                    {"ticker": "GOOG", "thesis": "Durable compounding thesis."}
                ],
            }
        )
        run = run_monitor(config=config, provider=SampleMarketDataProvider())

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = write_monitor_report(run, Path(tmpdir) / "report.html")

            self.assertTrue(report_path.exists())
            self.assertIn("Write report portfolio", report_path.read_text())

    def test_load_config_and_write_ledger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "portfolio.json"
            config_path.write_text(
                """
                {
                  "name": "File portfolio",
                  "positions": [
                    {"ticker": "GOOG", "thesis": "Durable compounding thesis."}
                  ]
                }
                """
            )
            config = load_portfolio_config(config_path)
            run = run_monitor(config=config, provider=SampleMarketDataProvider())

            ledger_path = write_ledger(run, Path(tmpdir) / "runs")

            self.assertTrue(ledger_path.exists())
            self.assertIn(run.run_id, ledger_path.read_text())


if __name__ == "__main__":
    unittest.main()
