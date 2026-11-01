# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""网络层：绕开坏代理，并给 requests 默认超时，避免单源卡死整次 fetch。"""

from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

_PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

# 默认直连。需要走代理时设 FLOW_MONITOR_NO_PROXY=0。
_DEFAULT_NO_PROXY = "1"
# 单次 HTTP 默认超时（秒）。金十分页等慢源可自行传更大 timeout。
_DEFAULT_TIMEOUT = 15.0

_prepared = False
_cleared: dict[str, str] = {}
_requests_patched = False
_orig_request = None


def _truthy(raw: str) -> bool:
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_has_proxy() -> bool:
    return any(os.environ.get(key) for key in _PROXY_KEYS)


def is_proxy_error(exc: BaseException) -> bool:
    """识别 requests/urllib 的代理失败，便于立刻改直连而不是逐条空转。"""
    text = f"{type(exc).__name__} {exc}".lower()
    return "proxy" in text or "proxyerror" in type(exc).__name__.lower()


def disable_env_proxy() -> dict[str, str]:
    """清掉进程内代理变量，并让 NO_PROXY=* 覆盖残留会话。"""
    global _cleared
    saved: dict[str, str] = {}
    for key in _PROXY_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    _cleared = saved
    if saved:
        _log.warning("已关闭环境代理并改为直连：%s", ", ".join(sorted(saved)))
    return saved


def patch_requests_timeout(timeout: float = _DEFAULT_TIMEOUT) -> None:
    """给 Session.request 补默认 timeout，避免东财断连时长时间挂起。"""
    global _requests_patched, _orig_request
    if _requests_patched:
        return
    try:
        import requests
    except ImportError:
        return

    _orig_request = requests.sessions.Session.request

    def _request(self: Any, method: str, url: str, **kwargs: Any):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return _orig_request(self, method, url, **kwargs)

    requests.sessions.Session.request = _request  # type: ignore[method-assign]
    _requests_patched = True
    _log.info("requests 默认 timeout=%ss", timeout)


def prepare_network() -> None:
    """fetch 开始时调用一次。默认绕开坏代理，并打上请求超时。"""
    global _prepared
    if _prepared:
        return
    _prepared = True
    raw = os.getenv("FLOW_MONITOR_NO_PROXY", _DEFAULT_NO_PROXY)
    if _truthy(raw) and env_has_proxy():
        disable_env_proxy()
    timeout_raw = os.getenv("FLOW_MONITOR_HTTP_TIMEOUT", "").strip()
    timeout = float(timeout_raw) if timeout_raw else _DEFAULT_TIMEOUT
    patch_requests_timeout(timeout)


def reset_network_state_for_tests() -> None:
    """测试用：恢复 prepare 开关；若已 patch 则还原 requests。"""
    global _prepared, _cleared, _requests_patched, _orig_request
    _prepared = False
    _cleared = {}
    if _requests_patched and _orig_request is not None:
        import requests

        requests.sessions.Session.request = _orig_request  # type: ignore[method-assign]
        _requests_patched = False
        _orig_request = None
