#!/usr/bin/env python3
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE / "forward_parts"
source = "".join((PARTS / f"{i:02d}.part").read_text(encoding="utf-8") for i in range(1, 6))
exec(
    compile(source, str(HERE / "r10_forward_live.generated.py"), "exec"),
    {"__name__": "__main__", "__file__": str(HERE / "r10_forward_live.generated.py")},
)
