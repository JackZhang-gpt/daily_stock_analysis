# -*- coding: utf-8 -*-
"""
Integrated research workflow adapted from findata-toolkit-cn.

Builds structured factor / diagnostic / signal context so the existing
daily_stock_analysis pipeline can produce richer outputs without replacing
its current trend-first architecture.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


WORKFLOW_STAGE_WEIGHTS: Dict[str, float] = {
    "technical": 0.35,
    "factors": 0.20,
    "quality": 0.15,
    "valuation": 0.10,
    "event": 0.10,
    "rotation": 0.10,
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(parsed):
            return None
        return parsed
    text = str(value).strip().replace(",", "").replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _clip_score(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _score_inverse(value: Optional[float], *, best: float, okay: float, weak: float) -> Optional[float]:
    if value is None:
        return None
    if value <= best:
        return 90.0
    if value <= okay:
        return 70.0
    if value <= weak:
        return 45.0
    return 20.0


def _score_positive(value: Optional[float], *, strong: float, good: float, neutral: float) -> Optional[float]:
    if value is None:
        return None
    if value >= strong:
        return 90.0
    if value >= good:
        return 70.0
    if value >= neutral:
        return 50.0
    return 25.0


def _keyword_flag(text: str, keywords: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _normalize_board_names(raw_boards: Any) -> List[str]:
    if not isinstance(raw_boards, list):
        return []
    names: List[str] = []
    for item in raw_boards:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _extract_block(ctx: Optional[Dict[str, Any]], block: str) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return {}
    payload = ctx.get(block, {})
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data", {})
    return data if isinstance(data, dict) else {}


def _compute_returns(history_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    if history_df is None or history_df.empty or "close" not in history_df.columns:
        return {"ret_20d": None, "ret_60d": None, "ret_120d": None, "volatility_20d": None}
    closes = pd.to_numeric(history_df["close"], errors="coerce").dropna().reset_index(drop=True)
    if closes.empty:
        return {"ret_20d": None, "ret_60d": None, "ret_120d": None, "volatility_20d": None}

    def _ret(period: int) -> Optional[float]:
        if len(closes) <= period:
            return None
        base = closes.iloc[-period - 1]
        if base == 0 or pd.isna(base):
            return None
        return (closes.iloc[-1] / base - 1.0) * 100.0

    returns = closes.pct_change().dropna()
    volatility = None
    if len(returns) >= 20:
        volatility = float(returns.tail(20).std() * math.sqrt(252.0) * 100.0)
    return {
        "ret_20d": _ret(20),
        "ret_60d": _ret(60),
        "ret_120d": _ret(120),
        "volatility_20d": volatility,
    }


def build_research_framework(
    *,
    stock_code: str,
    stock_name: str,
    history_df: Optional[pd.DataFrame],
    trend_result: Any,
    realtime_quote: Any,
    fundamental_context: Optional[Dict[str, Any]],
    belong_boards: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a structured factor / workflow payload for prompts and agents."""
    valuation = _extract_block(fundamental_context, "valuation")
    growth = _extract_block(fundamental_context, "growth")
    earnings = _extract_block(fundamental_context, "earnings")
    institution = _extract_block(fundamental_context, "institution")
    capital_flow = _extract_block(fundamental_context, "capital_flow")
    dragon_tiger = _extract_block(fundamental_context, "dragon_tiger")
    boards = _extract_block(fundamental_context, "boards")

    returns = _compute_returns(history_df)
    pe_ratio = _safe_float(valuation.get("pe_ratio"))
    pb_ratio = _safe_float(valuation.get("pb_ratio"))
    total_mv = _safe_float(valuation.get("total_mv"))
    dividend_yield = _safe_float((earnings.get("dividend") or {}).get("ttm_dividend_yield_pct"))

    revenue_yoy = _safe_float(growth.get("revenue_yoy"))
    profit_yoy = _safe_float(growth.get("net_profit_yoy"))
    roe = _safe_float(growth.get("roe"))
    roa = _safe_float(growth.get("roa"))
    gross_margin = _safe_float(growth.get("gross_margin"))
    net_margin = _safe_float(growth.get("net_margin"))
    debt_ratio = _safe_float(growth.get("debt_ratio"))
    current_ratio = _safe_float(growth.get("current_ratio"))
    quick_ratio = _safe_float(growth.get("quick_ratio"))
    asset_turnover = _safe_float(growth.get("asset_turnover"))

    financial_report = earnings.get("financial_report") if isinstance(earnings.get("financial_report"), dict) else {}
    revenue = _safe_float(financial_report.get("revenue"))
    net_profit_parent = _safe_float(financial_report.get("net_profit_parent"))
    operating_cash_flow = _safe_float(financial_report.get("operating_cash_flow"))

    turnover_rate = _safe_float(getattr(realtime_quote, "turnover_rate", None))
    price = _safe_float(getattr(realtime_quote, "price", None))
    atr_pct = None
    if history_df is not None and not history_df.empty and {"high", "low", "close"}.issubset(history_df.columns):
        d = history_df.copy()
        d["high"] = pd.to_numeric(d["high"], errors="coerce")
        d["low"] = pd.to_numeric(d["low"], errors="coerce")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        tr = pd.concat(
            [
                d["high"] - d["low"],
                (d["high"] - d["close"].shift()).abs(),
                (d["low"] - d["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1] if len(d) >= 14 else None
        if atr14 is not None and price:
            atr_pct = float(atr14 / price * 100.0)

    value_score = _avg(
        [
            _score_inverse(pe_ratio, best=12.0, okay=25.0, weak=40.0),
            _score_inverse(pb_ratio, best=1.5, okay=3.0, weak=5.0),
            _score_positive(dividend_yield, strong=4.0, good=2.5, neutral=1.0),
        ]
    )
    momentum_score = _avg(
        [
            _score_positive(returns.get("ret_20d"), strong=25.0, good=10.0, neutral=0.0),
            _score_positive(returns.get("ret_60d"), strong=35.0, good=15.0, neutral=0.0),
            _score_positive(getattr(trend_result, "trend_strength", None), strong=80.0, good=65.0, neutral=50.0),
        ]
    )
    quality_score = _avg(
        [
            _score_positive(roe, strong=18.0, good=12.0, neutral=8.0),
            _score_positive(roa, strong=8.0, good=5.0, neutral=2.0),
            _score_positive(gross_margin, strong=35.0, good=25.0, neutral=15.0),
            _score_inverse(debt_ratio, best=35.0, okay=50.0, weak=65.0),
            _score_positive(current_ratio, strong=1.6, good=1.2, neutral=1.0),
        ]
    )
    low_vol_score = _avg(
        [
            _score_inverse(returns.get("volatility_20d"), best=18.0, okay=28.0, weak=40.0),
            _score_inverse(atr_pct, best=2.5, okay=4.5, weak=7.0),
        ]
    )
    size_score = None
    if total_mv is not None and total_mv > 0:
        mv_yi = total_mv / 1e8
        size_score = _score_inverse(mv_yi, best=80.0, okay=250.0, weak=800.0)
    growth_score = _avg(
        [
            _score_positive(revenue_yoy, strong=30.0, good=15.0, neutral=5.0),
            _score_positive(profit_yoy, strong=30.0, good=15.0, neutral=5.0),
            _score_positive(net_margin, strong=15.0, good=8.0, neutral=3.0),
        ]
    )

    factor_scores = {
        "value": _clip_score(value_score),
        "momentum": _clip_score(momentum_score),
        "quality": _clip_score(quality_score),
        "low_volatility": _clip_score(low_vol_score),
        "size": _clip_score(size_score),
        "growth": _clip_score(growth_score),
    }
    composite_factor_score = _avg(factor_scores.values())

    cash_conversion = None
    if operating_cash_flow is not None and net_profit_parent not in (None, 0):
        cash_conversion = operating_cash_flow / net_profit_parent

    rule_of_40 = None
    if revenue_yoy is not None and net_margin is not None:
        rule_of_40 = revenue_yoy + net_margin

    piotroski_proxy = 0
    proxy_flags = {
        "positive_profit": net_profit_parent is not None and net_profit_parent > 0,
        "positive_cash_flow": operating_cash_flow is not None and operating_cash_flow > 0,
        "cash_gt_profit": (
            operating_cash_flow is not None
            and net_profit_parent is not None
            and operating_cash_flow > net_profit_parent
        ),
        "roe_positive": roe is not None and roe > 0,
        "margin_healthy": net_margin is not None and net_margin > 5,
        "gross_margin_healthy": gross_margin is not None and gross_margin > 20,
        "debt_controlled": debt_ratio is not None and debt_ratio < 60,
        "liquidity_ok": current_ratio is not None and current_ratio > 1,
        "growth_positive": revenue_yoy is not None and revenue_yoy > 0,
    }
    piotroski_proxy = sum(1 for ok in proxy_flags.values() if ok)

    altman_proxy = _avg(
        [
            _score_positive(current_ratio, strong=1.8, good=1.3, neutral=1.0),
            _score_inverse(debt_ratio, best=30.0, okay=45.0, weak=65.0),
            _score_positive(cash_conversion * 100 if cash_conversion is not None else None, strong=120.0, good=90.0, neutral=60.0),
        ]
    )

    valuation_flags: List[str] = []
    if pe_ratio is not None and pe_ratio > 60:
        valuation_flags.append("PE显著偏高")
    if pb_ratio is not None and pb_ratio > 6:
        valuation_flags.append("PB显著偏高")
    if dividend_yield is not None and dividend_yield < 1 and pe_ratio is not None and pe_ratio > 35:
        valuation_flags.append("低股息高估值组合")

    red_flags: List[str] = []
    if cash_conversion is not None and cash_conversion < 0.8:
        red_flags.append("经营现金流转化偏弱")
    if debt_ratio is not None and debt_ratio > 65:
        red_flags.append("负债率偏高")
    if quick_ratio is not None and quick_ratio < 0.8:
        red_flags.append("速动比率偏弱")
    if net_margin is not None and net_margin < 3:
        red_flags.append("净利率偏低")

    strengths: List[str] = []
    if roe is not None and roe >= 15:
        strengths.append("ROE处于较强区间")
    if cash_conversion is not None and cash_conversion >= 1.0:
        strengths.append("盈利现金化较好")
    if rule_of_40 is not None and rule_of_40 >= 40:
        strengths.append("40法则达标")
    if piotroski_proxy >= 6:
        strengths.append("财务健康代理分较高")

    event_score = 50.0
    event_notes: List[str] = []
    recent_count = _safe_float(dragon_tiger.get("recent_count"))
    if recent_count:
        event_score += min(10.0, recent_count * 2.0)
        event_notes.append(f"近20日龙虎榜出现 {int(recent_count)} 次")
    inst_change = _safe_float(institution.get("institution_holding_change"))
    if inst_change is not None:
        if inst_change > 0:
            event_score += 8.0
            event_notes.append("机构持股变化偏正面")
        elif inst_change < 0:
            event_score -= 8.0
            event_notes.append("机构持股变化偏负面")
    top10_change = _safe_float(institution.get("top10_holder_change"))
    if top10_change is not None:
        if top10_change > 0:
            event_score += 5.0
        elif top10_change < 0:
            event_score -= 5.0
    forecast_text = str(earnings.get("forecast_summary") or "")
    if _keyword_flag(forecast_text, ["预增", "增长", "扭亏", "向好"]):
        event_score += 8.0
        event_notes.append("业绩预告偏正面")
    if _keyword_flag(forecast_text, ["预亏", "下滑", "首亏", "亏损"]):
        event_score -= 12.0
        event_notes.append("业绩预告偏负面")
    if dividend_yield is not None and dividend_yield >= 3:
        event_score += 4.0
        event_notes.append("TTM股息率具备支撑")
    event_score = _clip_score(event_score)

    board_names = _normalize_board_names(belong_boards)
    sector_rankings = boards.get("sector_rankings") if isinstance(boards.get("sector_rankings"), dict) else {}
    top_boards = sector_rankings.get("top") if isinstance(sector_rankings, dict) else []
    bottom_boards = sector_rankings.get("bottom") if isinstance(sector_rankings, dict) else []
    rotation_score = 50.0
    rotation_notes: List[str] = []
    matched_top: List[str] = []
    matched_bottom: List[str] = []
    for item in top_boards or []:
        name = str(item.get("name") or "").strip()
        if name and any(name in board or board in name for board in board_names):
            matched_top.append(name)
    for item in bottom_boards or []:
        name = str(item.get("name") or "").strip()
        if name and any(name in board or board in name for board in board_names):
            matched_bottom.append(name)
    if matched_top:
        rotation_score += 20.0
        rotation_notes.append(f"所属板块位于涨幅榜：{', '.join(matched_top[:3])}")
    if matched_bottom:
        rotation_score -= 20.0
        rotation_notes.append(f"所属板块位于跌幅榜：{', '.join(matched_bottom[:3])}")
    main_flow = _safe_float(capital_flow.get("main_net_inflow"))
    if main_flow is not None:
        if main_flow > 0:
            rotation_score += 6.0
            rotation_notes.append("主力净流入为正")
        elif main_flow < 0:
            rotation_score -= 6.0
            rotation_notes.append("主力净流入为负")
    rotation_score = _clip_score(rotation_score)

    sentiment_gap_score = 50.0
    sentiment_gap_reason = "情绪与基本面大致匹配"
    if returns.get("ret_60d") is not None and returns["ret_60d"] < -15 and quality_score and quality_score >= 65:
        sentiment_gap_score = 78.0
        sentiment_gap_reason = "价格承压但质量分较高，存在情绪偏差修复可能"
    elif returns.get("ret_20d") is not None and returns["ret_20d"] > 35 and quality_score and quality_score < 55:
        sentiment_gap_score = 28.0
        sentiment_gap_reason = "短期价格显著超前于质量因子，需警惕透支"
    elif returns.get("ret_20d") is not None and returns["ret_20d"] < 0 and growth_score and growth_score >= 65:
        sentiment_gap_score = 68.0
        sentiment_gap_reason = "成长基本面优于价格表现，具备逆向观察价值"

    technical_score = _clip_score(_safe_float(getattr(trend_result, "signal_score", None))) or 50.0
    quality_stage_score = _clip_score(_avg([quality_score, altman_proxy, float(piotroski_proxy) / 9.0 * 100.0]))
    valuation_stage_score = _clip_score(_avg([value_score, 100.0 - len(valuation_flags) * 15.0]))
    factor_stage_score = _clip_score(composite_factor_score)
    event_stage_score = event_score
    rotation_stage_score = _avg([rotation_score, float(sentiment_gap_score)])

    workflow_score = 0.0
    workflow_breakdown: Dict[str, Optional[float]] = {
        "technical": technical_score,
        "factors": factor_stage_score,
        "quality": quality_stage_score,
        "valuation": valuation_stage_score,
        "event": event_stage_score,
        "rotation": rotation_stage_score,
    }
    total_weight = 0.0
    for key, weight in WORKFLOW_STAGE_WEIGHTS.items():
        score = workflow_breakdown.get(key)
        if score is None:
            continue
        workflow_score += score * weight
        total_weight += weight
    workflow_score = round(workflow_score / total_weight, 2) if total_weight else 50.0

    return {
        "identity": {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "belong_boards": board_names,
        },
        "workflow": {
            "stage_weights": {k: round(v, 2) for k, v in WORKFLOW_STAGE_WEIGHTS.items()},
            "stage_scores": {k: round(v, 2) if v is not None else None for k, v in workflow_breakdown.items()},
            "workflow_score": workflow_score,
            "stage_order": [
                "technical",
                "factors",
                "quality",
                "valuation",
                "event",
                "rotation",
            ],
            "method_note": (
                "workflow adapted from findata-toolkit-cn and merged into daily_stock_analysis; "
                "trend-first scoring is kept, then factor/quality/valuation/event/rotation are overlaid."
            ),
        },
        "factor_lens": {
            "composite_score": round(composite_factor_score, 2) if composite_factor_score is not None else None,
            "scores": {k: round(v, 2) if v is not None else None for k, v in factor_scores.items()},
            "timing_hint": (
                "复苏/扩张阶段偏重动量与成长，波动加大时回到质量与低波。"
            ),
        },
        "diagnostic_lens": {
            "cash_conversion_ratio": round(cash_conversion, 3) if cash_conversion is not None else None,
            "rule_of_40": round(rule_of_40, 2) if rule_of_40 is not None else None,
            "piotroski_proxy": piotroski_proxy,
            "altman_proxy_score": round(altman_proxy, 2) if altman_proxy is not None else None,
            "dupont_proxy": {
                "roe": roe,
                "net_margin": net_margin,
                "asset_turnover": asset_turnover,
                "debt_ratio": debt_ratio,
            },
            "strengths": strengths,
            "red_flags": red_flags,
        },
        "valuation_lens": {
            "pe_ratio": pe_ratio,
            "pb_ratio": pb_ratio,
            "dividend_yield_pct": dividend_yield,
            "market_cap_yi": round(total_mv / 1e8, 2) if total_mv is not None else None,
            "valuation_flags": valuation_flags,
            "tech_framework_hint": (
                "高增长科技股优先看PS/PEG/40法则，成熟盈利股优先看PE/FCF/分红。"
            ),
        },
        "signal_lens": {
            "sentiment_gap": {
                "score": round(float(sentiment_gap_score), 2),
                "reason": sentiment_gap_reason,
            },
            "event_driven": {
                "score": round(event_score, 2) if event_score is not None else None,
                "notes": event_notes,
                "dragon_tiger": {
                    "is_on_list": bool(dragon_tiger.get("is_on_list")),
                    "recent_count": dragon_tiger.get("recent_count"),
                    "latest_date": dragon_tiger.get("latest_date"),
                },
            },
            "sector_rotation": {
                "score": round(rotation_score, 2) if rotation_score is not None else None,
                "notes": rotation_notes,
                "matched_top": matched_top,
                "matched_bottom": matched_bottom,
            },
        },
        "raw_metrics": {
            "revenue_yoy": revenue_yoy,
            "net_profit_yoy": profit_yoy,
            "roe": roe,
            "roa": roa,
            "gross_margin": gross_margin,
            "net_margin": net_margin,
            "debt_ratio": debt_ratio,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "turnover_rate": turnover_rate,
            "ret_20d": returns.get("ret_20d"),
            "ret_60d": returns.get("ret_60d"),
            "ret_120d": returns.get("ret_120d"),
            "volatility_20d": returns.get("volatility_20d"),
            "atr_pct": atr_pct,
        },
    }
