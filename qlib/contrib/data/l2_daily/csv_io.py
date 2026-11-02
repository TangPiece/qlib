"""万得 L2 CSV 读取与列名规范化。"""
from __future__ import annotations

from io import BytesIO

import pandas as pd


def _depth_cols(prefix: str) -> list[str]:
    """生成 10 档盘口列名。"""
    return [f"{prefix}{i}" for i in range(1, 11)]


NUMERIC_COLUMNS: dict[str, list[str]] = {
    "逐笔成交": ["成交价格", "成交数量"],
    "逐笔委托": ["委托价格", "委托数量"],
    "行情": [
        "成交价",
        "成交量",
        "成交额",
        "成交笔数",
        "当日累计成交量",
        "当日成交额",
        "最高价",
        "最低价",
        "开盘价",
        "前收盘",
        *_depth_cols("申卖价"),
        *_depth_cols("申卖量"),
        *_depth_cols("申买价"),
        *_depth_cols("申买量"),
        "加权平均叫卖价",
        "加权平均叫买价",
        "叫卖总量",
        "叫买总量",
        "不加权指数",
        "品种总数",
        "上涨品种数",
        "下跌品种数",
        "持平品种数",
    ],
}


def normalize(df: pd.DataFrame, ftype: str) -> pd.DataFrame:
    """忠实规范化：去 Unnamed、strip 列名、时间补零、数值定型。

    2018 年部分万得导出逗号后带空格，列名会变成「 时间」；不 strip 则 prepare_trades 得到 0 行。
    """
    frame = df.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = frame[[c for c in frame.columns if not str(c).startswith("Unnamed")]]
    if "时间" in frame.columns:
        frame["时间"] = frame["时间"].astype(str).str.zfill(9)
    for col in NUMERIC_COLUMNS.get(ftype, []):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


# 同日可能混用 GBK/GB18030 与带 BOM 的 UTF-8（行情 vs 逐笔），按此顺序回退。
_CSV_ENCODINGS = ("gb18030", "gbk", "utf-8-sig", "utf-8")


def read_csv_bytes(csv_bytes: bytes, ftype: str) -> pd.DataFrame:
    """读入万得 CSV 并规范化；编码按 gb18030→gbk→utf-8-sig→utf-8 回退。"""
    last_err: UnicodeDecodeError | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            frame = pd.read_csv(
                BytesIO(csv_bytes),
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                low_memory=False,
            )
            return normalize(frame, ftype)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    tried = ", ".join(_CSV_ENCODINGS)
    sample = csv_bytes[:1] if csv_bytes else b"\x00"
    raise UnicodeDecodeError(
        "csv",
        sample,
        0,
        1,
        f"all encodings failed ({tried}): {last_err}",
    ) from last_err
