"""Fail-closed preflight for AlphaPilot V2 real industry/broker data.

No proxy variables are permitted. This script never prints credentials.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests

OUT = Path("artifacts/v2_real_data_preflight")
OUT.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ.get("FINMIND_TOKEN", "").strip()
API = "https://api.finmindtrade.com/api/v4/data"


def query(dataset: str, **params) -> pd.DataFrame:
    if not TOKEN:
        raise RuntimeError(
            "FINMIND_TOKEN is absent. Historical broker-branch and industry-chain "
            "datasets require an authorized sponsor token; proxy substitution is forbidden."
        )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    q = {"dataset": dataset, **params}
    r = requests.get(API, headers=headers, params=q, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("status") not in (200, None):
        raise RuntimeError(f"{dataset} rejected: {body.get('msg') or body.get('status')}")
    frame = pd.DataFrame(body.get("data", []))
    if frame.empty:
        raise RuntimeError(f"{dataset} returned no rows for {params}")
    return frame


def main() -> None:
    checks = {
        "token_present": bool(TOKEN),
        "broker_branch_real_not_proxy": False,
        "industry_money_flow_real_not_proxy": False,
        "historical_2021_access": False,
        "recent_2026_access": False,
        "formal_backtest_allowed": False,
    }
    details = {}
    try:
        broker_2021 = query(
            "TaiwanStockTradingDailyReport",
            data_id="2330",
            start_date="2021-01-04",
            end_date="2021-01-09",
        )
        broker_2026 = query(
            "TaiwanStockTradingDailyReport",
            data_id="2330",
            start_date="2026-08-01",
            end_date="2026-08-08",
        )
        industry_2021 = query(
            "TaiwanStockIndustryChainMoneyFlow",
            start_date="2021-01-04",
            end_date="2021-01-09",
        )
        industry_2026 = query(
            "TaiwanStockIndustryChainMoneyFlow",
            start_date="2026-08-01",
            end_date="2026-08-08",
        )
        required_broker = {"date", "stock_id"}
        required_industry = {"date"}
        checks["broker_branch_real_not_proxy"] = required_broker.issubset(broker_2021.columns)
        checks["industry_money_flow_real_not_proxy"] = required_industry.issubset(industry_2021.columns)
        checks["historical_2021_access"] = not broker_2021.empty and not industry_2021.empty
        checks["recent_2026_access"] = not broker_2026.empty and not industry_2026.empty
        details = {
            "broker_2021_rows": len(broker_2021),
            "broker_2026_rows": len(broker_2026),
            "industry_2021_rows": len(industry_2021),
            "industry_2026_rows": len(industry_2026),
            "broker_columns": list(broker_2021.columns),
            "industry_columns": list(industry_2021.columns),
        }
        checks["formal_backtest_allowed"] = all(
            checks[k]
            for k in (
                "token_present",
                "broker_branch_real_not_proxy",
                "industry_money_flow_real_not_proxy",
                "historical_2021_access",
                "recent_2026_access",
            )
        )
    except Exception as exc:
        details["error"] = str(exc)
    report = {
        "policy": "FAIL CLOSED: no market-breadth or institutional-volume proxies",
        "checks": checks,
        "details": details,
    }
    (OUT / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not checks["formal_backtest_allowed"]:
        raise SystemExit("V2 formal backtest blocked: real-data preflight failed")


if __name__ == "__main__":
    main()
