#!/usr/bin/env python3
from __future__ import annotations

import io
from typing import Iterable

import pandas as pd

import build_v6_context_layers_resumable as ctx
import build_v6_stage0_company_layer_parallel as lane


def _clean_col(c) -> str:
    if isinstance(c, tuple):
        vals = []
        for v in c:
            s = str(v).strip()
            if not s or s == "nan" or s.startswith("Unnamed"):
                continue
            vals.append(s)
        return vals[-1] if vals else str(c[-1]).strip()
    return str(c).strip()


def _pick(cols: Iterable[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()):
    for c in cols:
        s = str(c).replace(" ", "")
        if all(k in s for k in include) and not any(k in s for k in exclude):
            return c
    return None


def _extract_rows(df: pd.DataFrame, y: int, m: int, market: str):
    if df is None or df.empty:
        return []
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    cols = list(df.columns)

    code_col = _pick(cols, ("公司代號",))
    name_col = _pick(cols, ("公司名稱",))
    rev_col = _pick(cols, ("當月營收",), ("去年", "累計"))
    prev_col = _pick(cols, ("上月營收",), ("累計",))
    last_col = _pick(cols, ("去年當月營收",), ("累計",))
    mom_col = _pick(cols, ("上月比較增減",))
    yoy_col = _pick(cols, ("去年同月增減",))
    if code_col is None or rev_col is None:
        return []

    out = []
    for _, row in df.iterrows():
        c = ctx.code4_fixed(row.get(code_col))
        if not c:
            continue
        out.append(
            {
                "period_year": y,
                "period_month": m,
                "available_date": ctx.base.revenue_available_date(y, m),
                "market": "TWSE" if market == "sii" else "TPEX",
                "code": c,
                "name": str(row.get(name_col, "")) if name_col else "",
                "revenue_thousand": ctx.base.num(row.get(rev_col)),
                "prev_month_revenue_thousand": ctx.base.num(row.get(prev_col)) if prev_col else None,
                "last_year_month_revenue_thousand": ctx.base.num(row.get(last_col)) if last_col else None,
                "mops_mom_pct": ctx.base.num(row.get(mom_col)) if mom_col else None,
                "mops_yoy_pct": ctx.base.num(row.get(yoy_col)) if yoy_col else None,
            }
        )
    return out


def _decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "big5", "cp950"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            pass
    return content.decode("utf-8", "replace")


def fetch_revenue_month_archive(y: int, m: int, market: str):
    """Official MOPS historical archive, replacing retired/blocked mops.twse.com.tw/nas endpoint.

    One checkpoint unit remains (year, month, market). Prefer the compact archive CSV; if a
    historical month has no CSV representation, fall back to the official domestic/foreign
    archive HTML partitions (_0/_1) and combine them before checkpointing.
    """
    roc = y - 1911
    errors = []

    csv_url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc}_{m}.csv"
    try:
        r = ctx.base.get(csv_url, timeout=30, tries=2)
        text = _decode(r.content)
        df = pd.read_csv(io.StringIO(text))
        rows = _extract_rows(df, y, m, market)
        if rows:
            rows = list({r["code"]: r for r in rows}.values())
            print(f"[REV ARCHIVE CSV] {y}-{m:02d} {market} rows={len(rows)}", flush=True)
            return rows
        errors.append("csv parsed but yielded no company rows")
    except Exception as e:
        errors.append(f"csv={type(e).__name__}:{e}")

    rows = []
    for part in (0, 1):
        url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc}_{m}_{part}.html"
        try:
            r = ctx.base.get(url, timeout=30, tries=2)
            text = _decode(r.content)
            tables = pd.read_html(io.StringIO(text))
            part_rows = []
            for df in tables:
                part_rows.extend(_extract_rows(df, y, m, market))
            if part_rows:
                rows.extend(part_rows)
                print(f"[REV ARCHIVE HTML] {y}-{m:02d} {market} part={part} rows={len(part_rows)}", flush=True)
            else:
                errors.append(f"html_part_{part}=no_rows")
        except Exception as e:
            errors.append(f"html_part_{part}={type(e).__name__}:{e}")

    rows = list({r["code"]: r for r in rows}.values())
    if not rows:
        raise RuntimeError(
            f"MOPS official archive yielded no rows for {y}-{m:02d} {market}; "
            + " | ".join(errors[:6])
        )
    return rows


# Preserve the existing dedicated checkpoint layer. Only swap the network source behind it.
ctx._original_fetch_revenue_month = fetch_revenue_month_archive


if __name__ == "__main__":
    lane.main()
