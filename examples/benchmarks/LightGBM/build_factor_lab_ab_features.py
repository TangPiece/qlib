#!/usr/bin/env python3
"""从 Factor Lab quality CSV（或现成 mapping TSV）生成 FactorLabAB 特征 JSON。

默认读 alphalens 最近一次 run 的 A+B 名单，写出
``feature_sets/factor_lab_ab_113.json``，供 ``FactorLabAB`` handler 加载。

用法::

    python examples/benchmarks/LightGBM/build_factor_lab_ab_features.py
    python examples/benchmarks/LightGBM/build_factor_lab_ab_features.py \\
        --quality /path/to/factor_quality.csv \\
        --out examples/benchmarks/LightGBM/feature_sets/factor_lab_ab_113.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_LGBM_DIR = Path(__file__).resolve().parent
_QLIB_ROOT = _LGBM_DIR.parents[2]
_ALPHALENS_EXAMPLES = _QLIB_ROOT.parent / "alphalens" / "alphalens" / "examples"
_DEFAULT_QUALITY = (
    _ALPHALENS_EXAMPLES
    / "factor_lab_output"
    / "20261120_002406"
    / "factor_quality.csv"
)
_DEFAULT_MAPPING = (
    _ALPHALENS_EXAMPLES
    / "factor_lab_output"
    / "20261120_002406"
    / "ab_qlib_mapping.tsv"
)
_DEFAULT_OUT = _LGBM_DIR / "feature_sets" / "factor_lab_ab_113.json"

# 与 alphalens qlib_bridge / MOM12_1 handler 对齐
MOMENTUM_FACTORS: Dict[str, str] = {
    "MOM12": "$close / Ref($close, 252) - 1",
    "MOM12-1": "Ref($close, 21) / Ref($close, 252) - 1",
    "MOM6-1": "Ref($close, 21) / Ref($close, 126) - 1",
    "MOM3": "$close / Ref($close, 63) - 1",
    "REV5": "Ref($close, 5) / $close - 1",
    "REV10": "Ref($close, 10) / $close - 1",
    "REV20": "Ref($close, 20) / $close - 1",
}

ALPHA158_CONF = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {},
}


def _ensure_qlib_path() -> None:
    if str(_QLIB_ROOT) not in sys.path:
        sys.path.insert(0, str(_QLIB_ROOT))


def _alpha158_map() -> Dict[str, str]:
    _ensure_qlib_path()
    from qlib.contrib.data.loader import Alpha158DL

    fields, names = Alpha158DL.get_feature_config(ALPHA158_CONF)
    return dict(zip(names, fields))


def _alpha360_map() -> Dict[str, str]:
    _ensure_qlib_path()
    from qlib.contrib.data.loader import Alpha360DL

    fields, names = Alpha360DL.get_feature_config()
    return dict(zip(names, fields))


def resolve_expr(source: str, factor_id: str) -> str:
    """按 source 解析为 Qlib 表达式（field 类加 $ 前缀）。"""
    src = str(source).lower()
    fid = str(factor_id)
    if src == "alpha158":
        return _alpha158_map()[fid]
    if src == "alpha360":
        return _alpha360_map()[fid]
    if src == "l2":
        return "$" + fid if not fid.startswith("$") else fid
    if src == "fund_cn":
        return "$" + fid if not fid.startswith("$") else fid
    if src == "research":
        if fid not in MOMENTUM_FACTORS:
            raise KeyError("unknown research factor: %s" % fid)
        return MOMENTUM_FACTORS[fid]
    raise KeyError("unknown source: %s" % source)


def from_mapping_tsv(path: Path) -> Tuple[List[str], List[str], Dict[str, int]]:
    """从已有 mapping TSV 读 fields/names。"""
    m = pd.read_csv(path, sep="\t")
    fields: List[str] = []
    names: List[str] = []
    for _, r in m.iterrows():
        expr = str(r["qlib_expr_or_field"])
        if str(r.get("kind", "")) == "field" and not expr.startswith("$"):
            expr = "$" + expr
        fields.append(expr)
        names.append(str(r["factor_id"]))
    sources = m.groupby("source").size().astype(int).to_dict()
    return fields, names, sources


def orient_expr(expr: str, direction: int) -> str:
    """按冻结方向把表达式改成越大越看多。用乘法，不用一元负号。"""
    if direction == 1:
        return expr
    if direction == -1:
        return "(-1)*(%s)" % expr
    raise ValueError("direction must be +1 or -1, got %s" % direction)


def from_freeze(path: Path) -> Tuple[List[str], List[str], Dict[str, int], int]:
    """读 TRAIN 冻结名单。方向为 0 或缺失的因子不进入特征。"""
    payload = json.loads(path.read_text())
    factors = payload.get("factors") or []
    fields: List[str] = []
    names: List[str] = []
    sources: Dict[str, int] = {}
    skipped = 0
    for row in factors:
        direction = row.get("signal_direction")
        try:
            direction = int(direction)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if direction not in (1, -1):
            skipped += 1
            continue
        fid = str(row["factor_id"])
        src = str(row["source"])
        names.append(fid)
        fields.append(orient_expr(resolve_expr(src, fid), direction))
        sources[src] = sources.get(src, 0) + 1
    return fields, names, sources, skipped


def from_quality_csv(path: Path) -> Tuple[List[str], List[str], Dict[str, int]]:
    """筛 status∈{A,B}，按 quality 降序解析表达式。"""
    df = pd.read_csv(path)
    ab = df[df["status"].isin(["A", "B"])].copy()
    ab = ab.sort_values(["status", "quality"], ascending=[True, False])
    fields: List[str] = []
    names: List[str] = []
    for _, r in ab.iterrows():
        names.append(str(r["factor"]))
        fields.append(resolve_expr(str(r["source"]), str(r["factor"])))
    sources = ab.groupby("source").size().astype(int).to_dict()
    return fields, names, sources


def main() -> None:
    p = argparse.ArgumentParser(description="Build FactorLabAB feature JSON")
    p.add_argument("--quality", type=Path, default=None, help="factor_quality.csv")
    p.add_argument("--mapping", type=Path, default=None, help="ab_qlib_mapping.tsv")
    p.add_argument("--freeze", type=Path, default=None, help="TRAIN freeze_eligible.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--prefer-mapping",
        action="store_true",
        default=True,
        help="若 mapping 存在则优先用 TSV（默认开）",
    )
    args = p.parse_args()

    out = args.out or (_LGBM_DIR / "feature_sets" / "factor_lab_freeze_165.json" if args.freeze else _DEFAULT_OUT)
    extra = {}
    if args.freeze:
        # 冻结名单优先，不再回退到全样本 A+B
        fields, names, sources, skipped = from_freeze(args.freeze)
        source_run = args.freeze.parent.name
        note = "from_freeze:%s" % args.freeze
        extra = {
            "direction_source": "freeze_eligible.json",
            "n_direction_skipped": skipped,
        }
    else:
        mapping = args.mapping or _DEFAULT_MAPPING
        quality = args.quality or _DEFAULT_QUALITY
        if args.prefer_mapping and mapping.exists():
            fields, names, sources = from_mapping_tsv(mapping)
            source_run = mapping.parent.name
            note = "from_mapping:%s" % mapping
        elif quality.exists():
            fields, names, sources = from_quality_csv(quality)
            source_run = quality.parent.name
            note = "from_quality:%s" % quality
        else:
            raise SystemExit("neither mapping nor quality found: %s / %s" % (mapping, quality))

    payload = {
        "source_run": source_run,
        "note": note,
        "n_features": len(fields),
        "fields": fields,
        "names": names,
        "sources": sources,
        **extra,
    }
    args.out = out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("wrote %s  n=%d  sources=%s" % (args.out, len(fields), sources))


if __name__ == "__main__":
    main()
