"""Audit normalized data schema and causal cutoff behavior."""
from pathlib import Path
from clean_data_adapter import load_repo_daily_data


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    data = load_repo_daily_data(repo)
    latest = data.latest_date()
    assert latest is not None, "no normalized market data found"
    today = data.on(latest)
    assert today, "latest date has no rows"
    assert any(r.stock_id == "0050" and r.market == "TWSE" for r in today), "0050 missing"
    causal = data.as_of(latest)
    assert all(r.trade_date <= latest for r in causal)
    assert all(r.volume >= 0 and r.trading_value >= 0 for r in causal)
    assert all(r.market in {"TWSE", "TPEX"} for r in causal)
    print("CLEAN DATA AUDIT: PASS")
    print({"latest_date": latest, "latest_rows": len(today), "total_rows": len(causal)})


if __name__ == "__main__":
    main()
