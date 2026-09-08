#!/usr/bin/env python3
"""Execute the locked R10 MAX logic on 2016-2020 without changing strategy parameters."""
from pathlib import Path
import hashlib, json, re

root=Path(__file__).resolve().parent.parent
src=(root/"scripts"/"r10_max_formal.py").read_text(encoding="utf-8")
manifest=json.loads(Path("input_manifest.json").read_text(encoding="utf-8"))
if manifest.get("status")!="PASS":
    raise RuntimeError("validation inputs did not pass audit")

# Replace only dataset identity/window bindings. Strategy constants and logic remain byte-derived
# from locked commit 3728a0045ab77d65b1bd9c73fcbe942f6c0bc0d9.
block=re.compile(r"EXPECTED_INPUT_HASHES = \{.*?\n\}\n\nfor filename, expected in EXPECTED_INPUT_HASHES.items\(\):.*?raise RuntimeError\(f'input SHA mismatch: \{filename\}: \{actual\} != \{expected\}'\)\n",re.S)
replacement="""EXPECTED_INPUT_HASHES = {
    'institutional_2015_2020.parquet': MANIFEST_HASHES['institutional'],
    'ohlcv_2015.parquet': MANIFEST_HASHES['ohlcv_2015'],
    'ohlcv_2016.parquet': MANIFEST_HASHES['ohlcv_2016'],
    'ohlcv_2017.parquet': MANIFEST_HASHES['ohlcv_2017'],
    'ohlcv_2018.parquet': MANIFEST_HASHES['ohlcv_2018'],
    'ohlcv_2019.parquet': MANIFEST_HASHES['ohlcv_2019'],
    'ohlcv_2020.parquet': MANIFEST_HASHES['ohlcv_2020'],
}
for filename, expected in EXPECTED_INPUT_HASHES.items():
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'input SHA mismatch: {filename}: {actual} != {expected}')
"""
if not block.search(src): raise RuntimeError("locked hash block signature changed")
src=block.sub(replacement,src,count=1)
src=src.replace("for y in range(2020, 2026):","for y in range(2015, 2021):",1)
src=src.replace("official_corporate_actions_2020_2025.csv","official_corporate_actions_2015_2020.csv")
src=src.replace("ohlcv_causal_2020_2025.csv.gz","ohlcv_causal_2015_2020.csv.gz")
src=src.replace("institutional_2020_2025.parquet","institutional_2015_2020.parquet")
src=src.replace("eval_dates = [d for d in all_dates if 20210104 <= int(d) <= 20251231]",
                "eval_dates = [d for d in all_dates if 20160104 <= int(d) <= 20201231]")
src=src.replace("assert eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
                "assert eval_dates[0] == 20160104 and eval_dates[-1] == 20201231")
src=src.replace("for year in range(2021, 2026):","for year in range(2016, 2021):")
src=src.replace("'exact_evaluation_window': eval_dates[0] == 20210104 and eval_dates[-1] == 20251231",
                "'exact_evaluation_window': eval_dates[0] == 20160104 and eval_dates[-1] == 20201231")

h={"institutional":manifest["institutional"]["sha256"]}
for row in manifest["ohlcv"]:
    h[f"ohlcv_{row['year']}"]=row["sha256"]
prefix="MANIFEST_HASHES="+repr(h)+"\n"
generated=Path("r10_max_locked_2016_2020_generated.py")
generated.write_text(prefix+src,encoding="utf-8")
locked_hash=hashlib.sha256((root/"scripts"/"r10_max_formal.py").read_bytes()).hexdigest()
if locked_hash!="2fef3ba99b7c83b5db21e29c5f1c2abd8b3df77583840a025e6df937ff81c0d0":
    raise RuntimeError("locked engine SHA changed: "+locked_hash)
print("LOCKED_ENGINE_SHA PASS",locked_hash,flush=True)
exec(compile(generated.read_text(encoding="utf-8"),str(generated),"exec"),{"__name__":"__main__","__file__":str(generated)})
