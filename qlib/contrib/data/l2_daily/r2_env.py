"""从 qlib 仓库根目录的 .env 填充空的 R2_* 环境变量。不读 QuantDinger。"""
from __future__ import annotations

import os
from pathlib import Path

_LOADED: Path | None = None


def qlib_root() -> Path:
    """qlib 仓库根：contrib/data/l2_daily/r2_env.py 往上四级。"""
    return Path(__file__).resolve().parents[4]


def env_file_path() -> Path:
    """R2_ENV_FILE 优先，否则 {qlib}/.env。"""
    raw = os.getenv("R2_ENV_FILE", "").strip() or os.getenv("LEVEL2_DOTENV", "").strip()
    if raw:
        return Path(raw)
    return qlib_root() / ".env"


def load_r2_env() -> Path | None:
    """load_dotenv(override=False)：只填当前为空的键。文件不存在返回 None。"""
    global _LOADED
    path = env_file_path()
    if not path.is_file():
        return None
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    _LOADED = path
    return path
