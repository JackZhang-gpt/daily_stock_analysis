# -*- coding: utf-8 -*-
"""Tests for integrated research framework and strategy base weights."""

import unittest
from types import SimpleNamespace

import pandas as pd

from src.core.research_framework import build_research_framework
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.strategies.aggregator import StrategyAggregator


class ResearchFrameworkTestCase(unittest.TestCase):
    def _make_history_df(self) -> pd.DataFrame:
        rows = []
        close = 10.0
        for idx in range(90):
            close *= 1.01
            rows.append(
                {
                    "date": f"2026-01-{(idx % 28) + 1:02d}",
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + idx * 10_000,
                }
            )
        return pd.DataFrame(rows)

    def test_build_research_framework_returns_expected_sections(self) -> None:
        history_df = self._make_history_df()
        trend_result = SimpleNamespace(signal_score=72, trend_strength=82)
        realtime_quote = SimpleNamespace(price=history_df["close"].iloc[-1], turnover_rate=4.5)
        fundamental_context = {
            "valuation": {"data": {"pe_ratio": 18, "pb_ratio": 2.2, "total_mv": 18_000_000_000}},
            "growth": {
                "data": {
                    "revenue_yoy": 22,
                    "net_profit_yoy": 18,
                    "roe": 16,
                    "roa": 7,
                    "gross_margin": 32,
                    "net_margin": 14,
                    "debt_ratio": 38,
                    "current_ratio": 1.5,
                    "quick_ratio": 1.1,
                    "asset_turnover": 0.72,
                }
            },
            "earnings": {
                "data": {
                    "financial_report": {
                        "revenue": 12_000_000_000,
                        "net_profit_parent": 1_800_000_000,
                        "operating_cash_flow": 2_100_000_000,
                    },
                    "dividend": {"ttm_dividend_yield_pct": 2.8},
                    "forecast_summary": "业绩预增，盈利持续改善",
                }
            },
            "institution": {"data": {"institution_holding_change": 1.2}},
            "capital_flow": {"data": {"main_net_inflow": 1000000}},
            "dragon_tiger": {"data": {"is_on_list": True, "recent_count": 2, "latest_date": "2026-05-01"}},
            "boards": {
                "data": {
                    "sector_rankings": {
                        "top": [{"name": "锂电池", "net_inflow": 1000000}],
                        "bottom": [],
                    }
                }
            },
        }

        framework = build_research_framework(
            stock_code="002460",
            stock_name="赣锋锂业",
            history_df=history_df,
            trend_result=trend_result,
            realtime_quote=realtime_quote,
            fundamental_context=fundamental_context,
            belong_boards=[{"name": "锂电池"}],
        )

        self.assertIn("workflow", framework)
        self.assertIn("factor_lens", framework)
        self.assertIn("diagnostic_lens", framework)
        self.assertIn("signal_lens", framework)
        self.assertGreater(framework["workflow"]["workflow_score"], 0)
        self.assertEqual(framework["signal_lens"]["sector_rotation"]["matched_top"], ["锂电池"])


class StrategyWeightTestCase(unittest.TestCase):
    def test_static_strategy_weight_affects_consensus(self) -> None:
        agg = StrategyAggregator()
        ctx = AgentContext()
        ctx.add_opinion(AgentOpinion(agent_name="strategy_bull_trend", signal="buy", confidence=0.6))
        ctx.add_opinion(AgentOpinion(agent_name="strategy_wave_theory", signal="sell", confidence=0.6))
        result = agg.aggregate(ctx)
        self.assertIsNotNone(result)
        # bull_trend has a higher static prior than wave_theory, so even when
        # the final bucket remains "hold", the weighted score should tilt above
        # the neutral midpoint of 3.0.
        self.assertGreater(result.raw_data.get("weighted_score", 0), 3.0)


if __name__ == "__main__":
    unittest.main()
