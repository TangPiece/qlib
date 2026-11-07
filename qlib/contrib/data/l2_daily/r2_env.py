"""从 qlib 仓库根目录的 .env 填充空的 R2_* 环境变量。不读 QuantDinger。"""
from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

_LOADED: Path | None = None
# 进程内只清一次代理，避免重复打日志。
_PROXY_PREPARED = False
_CLEARED_PROXY: dict[str, str] = {}

_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

# 默认直连。Clash 7890 挂掉时 boto3 会 ProxyConnectionError。
# 需要走代理时设 LEVEL2_L2_DAILY_NO_PROXY=0。
_DEFAULT_NO_PROXY = "1"


def _truthy(raw: str) -> bool:
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def r2_direct_enabled() -> bool:
    """默认绕开环境代理；LEVEL2_L2_DAILY_NO_PROXY=0 才走代理。"""
    return _truthy(os.getenv("LEVEL2_L2_DAILY_NO_PROXY", _DEFAULT_NO_PROXY))


def env_has_proxy() -> bool:
    return any(os.environ.get(key) for key in _PROXY_KEYS)


def disable_env_proxy() -> dict[str, str]:
    """清掉进程内代理变量，并让 NO_PROXY=* 覆盖残留会话。"""
    global _CLEARED_PROXY
    saved: dict[str, str] = {}
    for key in _PROXY_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    _CLEARED_PROXY = saved
    if saved:
        _log.warning("R2 已关闭环境代理并改为直连：%s", ", ".join(sorted(saved)))
    return saved


def prepare_r2_network() -> None:
    """默认直连 R2，避免本机代理挂掉时 put 失败。"""
    global _PROXY_PREPARED
    if _PROXY_PREPARED:
        return
    _PROXY_PREPARED = True
    if r2_direct_enabled() and env_has_proxy():
        disable_env_proxy()


def reset_r2_network_for_tests() -> None:
    """单测恢复代理相关状态，允许下一次 load_r2_env 再清一次。"""
    global _PROXY_PREPARED, _CLEARED_PROXY
    _PROXY_PREPARED = False
    _CLEARED_PROXY = {}


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
    """load_dotenv(override=False)：只填当前为空的键。随后按开关直连。

    文件不存在仍会 prepare 网络，返回 None。
    """
    global _LOADED
    path = env_file_path()
    result = None
    if path.is_file():
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        _LOADED = path
        result = path
    prepare_r2_network()
    return result
