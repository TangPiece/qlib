#!/usr/bin/env python3
"""兼容入口：控制台已迁至 examples/console。

Usage::

    .venv/bin/python examples/benchmarks/LightGBM/console/server.py
    # 等价于 examples/console/server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# LightGBM/console -> LightGBM -> benchmarks -> examples -> console/server.py
NEW_SERVER = Path(__file__).resolve().parents[3] / "console" / "server.py"


def main() -> None:
    if not NEW_SERVER.is_file():
        raise SystemExit(f"new console not found: {NEW_SERVER}")
    print("[shim] console moved to examples/console; forwarding…", flush=True)
    os.execv(sys.executable, [sys.executable, str(NEW_SERVER), *sys.argv[1:]])


if __name__ == "__main__":
    main()
