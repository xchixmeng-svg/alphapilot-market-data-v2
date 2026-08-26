import csv
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

VERSION = "AlphaPilot-Data-V3.7"
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
    return code if re.fullmatch(r"\d{4}", code) else None


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
norm_dir = base / "normalized"
raw_dir.mkdir(parents=True, exist_ok=True)
norm_dir.mkdir(parents=True, exist_ok=True)

raw_payloads = {
    "twse_ohlcv": twse_ohlcv_payload,
    "twse_ohlcv_snapshot": twse_snapshot_payload,
    "tpex_ohlcv": tpex_ohlcv_payload,
    "twse_institutional": twse_inst_payload,
    "tpex_institutional": tpex_inst_payload,
}
for name, payload in raw_payloads.items():
    (raw_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

normalized = {
    "twse_ohlcv": normalize_ohlcv(twse_ohlcv_rows, "TWSE"),
    "tpex_ohlcv": normalize_ohlcv(tpex_ohlcv_payload, "TPEX"),
    "twse_institutional": normalize_inst(twse_inst_rows, "TWSE"),
    "tpex_institutional": normalize_inst(tpex_inst_payload, "TPEX"),
}

for name, rows in normalized.items():
    write_csv(norm_dir / f"{name}.csv", rows)

errors = []
warnings = []
row_counts = {name: len(rows) for name, rows in normalized.items()}
minimum_rows = {
    "twse_ohlcv": 800,
    "tpex_ohlcv": 500,
    "twse_institutional": 700,
    "tpex_institutional": 400,
}
for name, minimum in minimum_rows.items():
    if row_counts.get(name, 0) < minimum:
        errors.append(
            f"{name}: row count {row_counts.get(name, 0)} < minimum {minimum}"
        )

for name, rows in normalized.items():
    codes = [row["stock_id"] for row in rows]
    duplicate_count = len(codes) - len(set(codes))
    if duplicate_count:
        errors.append(f"{name}: duplicate rows={duplicate_count}")

for name in ("twse_ohlcv", "tpex_ohlcv"):
    bad_ohlc = 0
    null_close = 0
    rows = normalized[name]
    for row in rows:
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        if c is None:
            null_close += 1
            continue
        if h is not None and l is not None and h < l:
            bad_ohlc += 1
        if h is not None and o is not None and h < o:
            bad_ohlc += 1
        if h is not None and c is not None and h < c:
            bad_ohlc += 1
        if l is not None and o is not None and l > o:
            bad_ohlc += 1
        if l is not None and c is not None and l > c:
            bad_ohlc += 1
    if bad_ohlc:
        errors.append(f"{name}: OHLC errors={bad_ohlc}")
    null_ratio = null_close / max(1, len(rows))
    if null_ratio > 0.05:
        errors.append(f"{name}: null close={null_ratio:.1%}")

coverage = {}
for market in ("twse", "tpex"):
    price_codes = {
        row["stock_id"] for row in normalized[f"{market}_ohlcv"]
    }
    inst_codes = {
        row["stock_id"] for row in normalized[f"{market}_institutional"]
    }
    overlap = len(price_codes & inst_codes)
    ratio = overlap / max(1, len(price_codes))
    coverage[market] = {
        "price_codes": len(price_codes),
        "institutional_codes": len(inst_codes),
        "overlap": overlap,
        "coverage": ratio,
    }
    if ratio < 0.65:
        errors.append(f"{market.upper()} institutional coverage={ratio:.1%}")

flattened_fields = {}
for name in ("twse_institutional", "tpex_institutional"):
    stats = {}
    for column in ("foreign_net", "trust_net", "dealer_net"):
        count = sum(
            1 for row in normalized[name] if row.get(column) is not None
        )
        stats[column] = count
        if count < 100:
            errors.append(f"{name}: {column} non-null only {count}")
    flattened_fields[name] = stats

source_dates = {
    "twse_ohlcv": trade_date,
    "tpex_ohlcv": str(tpex_ohlcv_date),
    "twse_institutional": trade_date,
    "tpex_institutional": str(tpex_inst_date),
}
if len(set(source_dates.values())) != 1:
    errors.append("final source date alignment failed")

if twse_snapshot_date != trade_day:
    warnings.append(
        "TWSE STOCK_DAY_ALL lagged target date; "
        "date-addressable MI_INDEX fallback was used."
    )

manifest = {
    "version": VERSION,
    "execution_time": now.isoformat(),
    "trade_date": trade_date,
    "status": "PASS" if not errors else "FAIL",
    "source_dates": source_dates,
    "row_counts": row_counts,
    "coverage": coverage,
    "flattened_fields": flattened_fields,
    "warnings": warnings,
    "errors": errors,
    "sources": {
        "twse_ohlcv": twse_ohlcv_source,
        "tpex_ohlcv": "TPEx tpex_mainboard_daily_close_quotes",
        "twse_institutional": "TWSE T86",
        "tpex_institutional": "TPEx tpex_3insti_daily_trading",
    },
}
(base / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

if errors:
    raise SystemExit(2)

Path(".alphapilot_trade_date").write_text(trade_date, encoding="utf-8")
