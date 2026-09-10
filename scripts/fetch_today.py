import csv
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

VERSION = "AlphaPilot-Data-V3.8"
TZ = ZoneInfo("Asia/Taipei")
now = datetime.now(TZ)

session = requests.Session()
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

TWSE_OPENAPI = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_OHLCV = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TPEX_INST = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"


def get_json(name, url, params=None):
    last_error = None
    for attempt in range(6):
        try:
            response = session.get(
                url,
                params=params or {},
                headers=HEADERS,
                timeout=(15, 60),
            )
            response.raise_for_status()
            if not response.text.strip():
                raise RuntimeError("empty response")
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                wait = min(30, 2 ** attempt)
                print(
                    f"[retry] {name} attempt={attempt + 1} "
                    f"error={exc} sleep={wait}s",
                    flush=True,
                )
                time.sleep(wait)
    raise RuntimeError(f"{name} failed after retries: {last_error}")


def norm_key(value):
    return re.sub(r"[\s_\-()/（）]+", "", str(value or "")).lower()


def num(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("+", "")
    if text in {"", "--", "---", "----", "null", "None"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def integer(value):
    value = num(value)
    return None if value is None else int(round(value))


def pick(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    normalized = {norm_key(k): v for k, v in mapping.items()}
    for key in keys:
        if norm_key(key) in normalized:
            return normalized[norm_key(key)]
    return None


def parse_date(value):
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value).strip())
    if len(digits) == 8:
        try:
            return datetime.strptime(digits, "%Y%m%d").date()
        except ValueError:
            pass
    if len(digits) == 7:
        try:
            year = int(digits[:3]) + 1911
            return datetime.strptime(f"{year}{digits[3:]}", "%Y%m%d").date()
        except ValueError:
            pass
    return None


DATE_KEYS = {
    "date",
    "tradedate",
    "tradingdate",
    "日期",
    "交易日期",
    "成交日期",
}


def collect_dates(payload, limit=1000):
    dates = []

    def walk(value):
        if len(dates) >= limit:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if norm_key(key) in DATE_KEYS:
                    parsed = parse_date(child)
                    if parsed:
                        dates.append(parsed)
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value[:500]:
                if isinstance(child, (dict, list)):
                    walk(child)
                if len(dates) >= limit:
                    return

    walk(payload)
    return dates


def payload_date(name, payload):
    dates = collect_dates(payload)
    if not dates:
        raise RuntimeError(f"{name}: cannot determine official payload date")
    counter = Counter(dates)
    best_date, count = counter.most_common(1)[0]
    ratio = count / len(dates)
    print(
        f"[date] {name}: {best_date} "
        f"votes={count}/{len(dates)} ratio={ratio:.1%}",
        flush=True,
    )
    if ratio < 0.80:
        raise RuntimeError(f"{name}: date consensus too weak ({ratio:.1%})")
    return best_date


def table_candidates(payload):
    candidates = []

    def add(fields, rows, title=""):
        if isinstance(fields, list) and isinstance(rows, list) and rows:
            candidates.append((fields, rows, str(title or "")))

    def walk(value):
        if isinstance(value, dict):
            fields = value.get("fields")
            rows = value.get("data")
            add(fields, rows, value.get("title", ""))
            for key, maybe_fields in value.items():
                if str(key).startswith("fields") and isinstance(maybe_fields, list):
                    suffix = str(key)[6:]
                    add(maybe_fields, value.get("data" + suffix), value.get("title", ""))
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, (dict, list)):
                    walk(child)

    walk(payload)
    return candidates


def find_table(payload, required_tokens, name):
    scored = []
    for fields, rows, title in table_candidates(payload):
        joined = "|".join(str(x) for x in fields)
        matched = sum(1 for token in required_tokens if token in joined)
        score = matched * 100000 + len(rows)
        if "證券代號" in joined:
            score += 10000
        if "證券名稱" in joined:
            score += 5000
        scored.append((score, matched, fields, rows, title))

    if not scored:
        raise RuntimeError(f"{name}: no table candidates found")

    scored.sort(reverse=True, key=lambda item: item[0])
    _, matched, fields, rows, title = scored[0]
    if matched < len(required_tokens):
        raise RuntimeError(
            f"{name}: best table missing required fields "
            f"matched={matched}/{len(required_tokens)} title={title!r}"
        )
    return fields, rows


def table_to_dicts(fields, rows):
    output = []
    for row in rows:
        if isinstance(row, list):
            output.append({
                fields[i]: row[i] if i < len(row) else None
                for i in range(len(fields))
            })
        elif isinstance(row, dict):
            output.append(row)
    return output


def stock_code(row):
    code = str(
        pick(
            row,
            "Code",
            "SecuritiesCompanyCode",
            "證券代號",
            "代號",
        )
        or ""
    ).strip()
    # Preserve ordinary four-digit stocks and TWSE ETF/ETN identifiers that
    # begin with 00, including six-character suffix codes such as 00631L.
    # R10 strategy-universe filtering remains separate and locked to four-digit equities.
    return code if re.fullmatch(r"(?:\d{4}|00[A-Z0-9]{4})", code, flags=re.IGNORECASE) else None


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matching_value(row, institutions, actions, reject_prefixes=()):
    candidates = []
    for key, value in row.items():
        normalized = norm_key(key)
        if any(normalized.startswith(norm_key(x)) for x in reject_prefixes):
            continue
        if not any(norm_key(x) in normalized for x in institutions):
            continue
        if not any(norm_key(x) in normalized for x in actions):
            continue
        candidates.append((-len(normalized), value))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def foreign_value(row, actions):
    value = matching_value(row, ["外陸資", "外資及陸資"], actions)
    if value is not None:
        return value
    return matching_value(
        row,
        ["Foreign", "外資"],
        actions,
        reject_prefixes=["外資自營商"],
    )


def trust_value(row, actions):
    return matching_value(
        row,
        ["InvestmentTrust", "Trust", "投信"],
        actions,
    )


def dealer_value(row, actions):
    candidates = []
    for key, value in row.items():
        normalized = norm_key(key)
        if normalized.startswith(norm_key("外資自營商")):
            continue
        if not (
            normalized.startswith(norm_key("自營商"))
            or "dealer" in normalized
        ):
            continue
        if not any(norm_key(x) in normalized for x in actions):
            continue
        score = -len(normalized)
        if "自行買賣" not in normalized and "避險" not in normalized:
            score += 10000
        candidates.append((score, value))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def normalize_ohlcv(rows, market):
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = stock_code(row)
        if not code:
            continue
        output.append({
            "trade_date": trade_date,
            "market": market,
            "stock_id": code,
            "name": pick(row, "Name", "CompanyName", "證券名稱", "名稱"),
            "open": num(pick(row, "OpeningPrice", "Open", "開盤價")),
            "high": num(pick(row, "HighestPrice", "High", "最高價")),
            "low": num(pick(row, "LowestPrice", "Low", "最低價")),
            "close": num(pick(row, "ClosingPrice", "Close", "收盤價")),
            "volume": integer(
                pick(row, "TradeVolume", "TradingShares", "成交股數", "成交量")
            ),
            "trading_value": integer(
                pick(row, "TradeValue", "TransactionAmount", "成交金額")
            ),
        })
    return output


def normalize_inst(rows, market):
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = stock_code(row)
        if not code:
            continue

        foreign_buy = integer(foreign_value(row, ["Buy", "買進"]))
        foreign_sell = integer(foreign_value(row, ["Sell", "賣出"]))
        foreign_net = integer(
            foreign_value(row, ["Net", "Difference", "買賣超", "差額"])
        )
        trust_buy = integer(trust_value(row, ["Buy", "買進"]))
        trust_sell = integer(trust_value(row, ["Sell", "賣出"]))
        trust_net = integer(
            trust_value(row, ["Net", "Difference", "買賣超", "差額"])
        )
        dealer_buy = integer(dealer_value(row, ["Buy", "買進"]))
        dealer_sell = integer(dealer_value(row, ["Sell", "賣出"]))
        dealer_net = integer(
            dealer_value(row, ["Net", "Difference", "買賣超", "差額"])
        )

        output.append({
            "trade_date": trade_date,
            "market": market,
            "stock_id": code,
            "name": pick(row, "Name", "CompanyName", "證券名稱", "名稱"),
            "foreign_buy": foreign_buy,
            "foreign_sell": foreign_sell,
            "foreign_net": foreign_net,
            "trust_buy": trust_buy,
            "trust_sell": trust_sell,
            "trust_net": trust_net,
            "dealer_buy": dealer_buy,
            "dealer_sell": dealer_sell,
            "dealer_net": dealer_net,
            "raw_json": json.dumps(
                row, ensure_ascii=False, separators=(",", ":")
            ),
        })
    return output


print("[fetch] tpex_ohlcv", flush=True)
tpex_ohlcv_payload = get_json("tpex_ohlcv", TPEX_OHLCV)

print("[fetch] tpex_institutional", flush=True)
tpex_inst_payload = get_json("tpex_institutional", TPEX_INST)

tpex_ohlcv_date = payload_date("tpex_ohlcv", tpex_ohlcv_payload)
tpex_inst_date = payload_date("tpex_institutional", tpex_inst_payload)

if tpex_ohlcv_date != tpex_inst_date:
    raise RuntimeError(
        "TPEx snapshot date mismatch: "
        f"ohlcv={tpex_ohlcv_date} institutional={tpex_inst_date}"
    )

trade_day = tpex_ohlcv_date
trade_date = trade_day.strftime("%Y-%m-%d")
ymd = trade_day.strftime("%Y%m%d")

print(f"[selected] target trade_date={trade_date}", flush=True)

print("[fetch] twse_ohlcv_snapshot", flush=True)
twse_snapshot_payload = get_json("twse_ohlcv_snapshot", TWSE_OPENAPI)
twse_snapshot_date = payload_date("twse_ohlcv_snapshot", twse_snapshot_payload)

twse_ohlcv_source = "TWSE STOCK_DAY_ALL"
twse_ohlcv_payload = twse_snapshot_payload
twse_ohlcv_rows = twse_snapshot_payload

if twse_snapshot_date != trade_day:
    print(
        "[fallback] TWSE STOCK_DAY_ALL is not aligned; "
        f"snapshot={twse_snapshot_date} target={trade_day}. "
        "Fetching date-addressable MI_INDEX.",
        flush=True,
    )
    twse_ohlcv_payload = get_json(
        "twse_ohlcv_mi_index",
        TWSE_MI_INDEX,
        {
            "date": ymd,
            "type": "ALLBUT0999",
            "response": "json",
        },
    )
    fields, rows = find_table(
        twse_ohlcv_payload,
        ["證券代號", "開盤價", "最高價", "最低價", "收盤價"],
        "twse_ohlcv_mi_index",
    )
    twse_ohlcv_rows = table_to_dicts(fields, rows)
    twse_ohlcv_source = "TWSE MI_INDEX fallback"

print(f"[fetch] twse_institutional date={ymd}", flush=True)
twse_inst_payload = get_json(
    "twse_institutional",
    TWSE_T86,
    {
        "date": ymd,
        "selectType": "ALLBUT0999",
        "response": "json",
    },
)
twse_inst_fields, twse_inst_table_rows = find_table(
    twse_inst_payload,
    ["證券代號", "外資", "投信", "自營商"],
    "twse_institutional",
)
twse_inst_rows = table_to_dicts(twse_inst_fields, twse_inst_table_rows)

base = Path("data") / trade_date
raw_dir = base / "raw"
normalized_dir = base / "normalized"
raw_dir.mkdir(parents=True, exist_ok=True)
normalized_dir.mkdir(parents=True, exist_ok=True)

(raw_dir / "tpex_ohlcv.json").write_text(
    json.dumps(tpex_ohlcv_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(raw_dir / "tpex_institutional.json").write_text(
    json.dumps(tpex_inst_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(raw_dir / "twse_ohlcv_snapshot.json").write_text(
    json.dumps(twse_snapshot_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(raw_dir / "twse_ohlcv.json").write_text(
    json.dumps(twse_ohlcv_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(raw_dir / "twse_institutional.json").write_text(
    json.dumps(twse_inst_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

twse_ohlcv = normalize_ohlcv(twse_ohlcv_rows, "TWSE")
tpex_ohlcv = normalize_ohlcv(tpex_ohlcv_payload, "TPEX")
twse_inst = normalize_inst(twse_inst_rows, "TWSE")
tpex_inst = normalize_inst(tpex_inst_payload, "TPEX")

for name, rows in [
    ("twse_ohlcv", twse_ohlcv),
    ("tpex_ohlcv", tpex_ohlcv),
    ("twse_institutional", twse_inst),
    ("tpex_institutional", tpex_inst),
]:
    if not rows:
        raise RuntimeError(f"{name}: normalized dataset is empty")
    if any(row["trade_date"] != trade_date for row in rows):
        raise RuntimeError(f"{name}: normalized trade_date mismatch")

write_csv(normalized_dir / "twse_ohlcv.csv", twse_ohlcv)
write_csv(normalized_dir / "tpex_ohlcv.csv", tpex_ohlcv)
write_csv(normalized_dir / "twse_institutional.csv", twse_inst)
write_csv(normalized_dir / "tpex_institutional.csv", tpex_inst)


def coverage(price_rows, inst_rows):
    price_codes = {row["stock_id"] for row in price_rows}
    inst_codes = {row["stock_id"] for row in inst_rows}
    overlap = price_codes & inst_codes
    ratio = len(overlap) / max(1, len(price_codes))
    return {
        "price_codes": len(price_codes),
        "institutional_codes": len(inst_codes),
        "overlap": len(overlap),
        "coverage": ratio,
    }


manifest = {
    "dataset_version": VERSION,
    "trade_date": trade_date,
    "generated_at": now.isoformat(),
    "status": "PASS",
    "sources": {
        "twse_ohlcv": twse_ohlcv_source,
        "twse_institutional": "TWSE T86",
        "tpex_ohlcv": "TPEx OpenAPI tpex_mainboard_daily_close_quotes",
        "tpex_institutional": "TPEx OpenAPI tpex_3insti_daily_trading",
    },
    "source_dates": {
        "twse_ohlcv": trade_date,
        "twse_institutional": trade_date,
        "tpex_ohlcv": trade_date,
        "tpex_institutional": trade_date,
    },
    "coverage": {
        "twse": coverage(twse_ohlcv, twse_inst),
        "tpex": coverage(tpex_ohlcv, tpex_inst),
    },
}

(base / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
