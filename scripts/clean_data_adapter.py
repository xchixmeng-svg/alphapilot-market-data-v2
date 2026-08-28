"""Causal adapter for AlphaPilot normalized TWSE/TPEx daily files."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

OHLCV_COLUMNS = {
    "trade_date", "market", "stock_id", "name",
    "open", "high", "low", "close", "volume", "trading_value",
}
INST_COLUMNS = {
    "trade_date", "market", "stock_id", "name",
    "foreign_buy", "foreign_sell", "foreign_net",
    "trust_buy", "trust_sell", "trust_net",
    "dealer_buy", "dealer_sell", "dealer_net",
}

Key = Tuple[str, str, str]


@dataclass(frozen=True)
class DailyRow:
    trade_date: str
    market: str
    stock_id: str
    name: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: int
    trading_value: int
    foreign_net: int = 0
    trust_net: int = 0
    dealer_net: int = 0


def _float(v: str) -> Optional[float]:
    s = (v or "").strip()
    return None if not s else float(s)


def _int(v: str) -> int:
    s = (v or "").strip().replace(",", "")
    return 0 if not s else int(float(s))


def _read_csv(path: Path, required: set[str]) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def load_normalized_day(day_dir: Path) -> List[DailyRow]:
    """Load one normalized/YYYY-MM-DD dataset without using future dates."""
    ohlcv: Dict[Key, dict] = {}
    inst: Dict[Key, dict] = {}
    for market in ("twse", "tpex"):
        op = day_dir / f"{market}_ohlcv.csv"
        ip = day_dir / f"{market}_institutional.csv"
        if op.exists():
            for r in _read_csv(op, OHLCV_COLUMNS):
                key = (r["trade_date"], r["market"], r["stock_id"])
                if key in ohlcv:
                    raise ValueError(f"duplicate OHLCV key: {key}")
                ohlcv[key] = r
        if ip.exists():
            for r in _read_csv(ip, INST_COLUMNS):
                key = (r["trade_date"], r["market"], r["stock_id"])
                if key in inst:
                    raise ValueError(f"duplicate institutional key: {key}")
                inst[key] = r

    rows: List[DailyRow] = []
    for key, r in ohlcv.items():
        i = inst.get(key, {})
        rows.append(DailyRow(
            trade_date=r["trade_date"], market=r["market"], stock_id=r["stock_id"],
            name=(r.get("name") or "").strip(), open=_float(r["open"]),
            high=_float(r["high"]), low=_float(r["low"]), close=_float(r["close"]),
            volume=_int(r["volume"]), trading_value=_int(r["trading_value"]),
            foreign_net=_int(i.get("foreign_net", "0")),
            trust_net=_int(i.get("trust_net", "0")),
            dealer_net=_int(i.get("dealer_net", "0")),
        ))
    rows.sort(key=lambda x: (x.trade_date, x.market, x.stock_id))
    return rows


class CausalMarketData:
    """In-memory daily data store whose public query is cutoff-safe."""
    def __init__(self, rows: Iterable[DailyRow]):
        self._rows = sorted(rows, key=lambda x: (x.trade_date, x.market, x.stock_id))

    def as_of(self, cutoff_date: str) -> List[DailyRow]:
        return [r for r in self._rows if r.trade_date <= cutoff_date]

    def on(self, trade_date: str) -> List[DailyRow]:
        return [r for r in self._rows if r.trade_date == trade_date]

    def latest_date(self) -> Optional[str]:
        return self._rows[-1].trade_date if self._rows else None


def load_repo_daily_data(repo_root: Path) -> CausalMarketData:
    rows: List[DailyRow] = []
    data_root = repo_root / "data"
    if not data_root.exists():
        return CausalMarketData([])
    for p in sorted(data_root.iterdir()):
        normalized = p / "normalized"
        if p.is_dir() and normalized.is_dir():
            rows.extend(load_normalized_day(normalized))
    return CausalMarketData(rows)
