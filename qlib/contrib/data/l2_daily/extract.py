"""用系统 tar 解压当日 .7z。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _normalize_member_path(raw: str) -> str:
    """去掉 tar 列表中的空白与 ./ 前缀。"""
    return raw.strip().lstrip("./")


def list_members(seven_z_path: str | Path) -> list[str]:
    """列出归档内 csv 成员路径。"""
    out = subprocess.check_output(["tar", "-tf", str(seven_z_path)], text=True)
    return [
        name
        for line in out.splitlines()
        if (name := _normalize_member_path(line)).endswith(".csv")
    ]


def _archive_has_date_prefix(seven_z_path: Path, date: str) -> bool:
    """归档成员是否以 {date}/ 为顶层。"""
    prefix = f"{date}/"
    members = list_members(seven_z_path)
    if members:
        return any(m.startswith(prefix) for m in members)
    out = subprocess.check_output(["tar", "-tf", str(seven_z_path)], text=True)
    for line in out.splitlines():
        name = _normalize_member_path(line)
        if not name:
            continue
        return name == date or name == f"{date}/" or name.startswith(prefix)
    return False


def _looks_like_code_dir(name: str) -> bool:
    return "." in name and not name.startswith(".")


def _is_nested_date_layout(day_dir: Path, date: str) -> bool:
    nested = day_dir / date
    if not nested.is_dir():
        return False
    nested_codes = [p for p in nested.iterdir() if p.is_dir() and _looks_like_code_dir(p.name)]
    if not nested_codes:
        return False
    top_codes = [
        p for p in day_dir.iterdir() if p.is_dir() and p.name != date and _looks_like_code_dir(p.name)
    ]
    return not top_codes


def _flatten_nested_date_dir(day_dir: Path, date: str) -> bool:
    """把 day_dir/{date}/* 上移，消除双层日期。"""
    if not _is_nested_date_layout(day_dir, date):
        return False
    nested = day_dir / date
    print(f"  展平双层日期目录: {nested} → {day_dir}", flush=True)
    for item in list(nested.iterdir()):
        dest = day_dir / item.name
        if dest.exists():
            raise FileExistsError(f"展平冲突：目标已存在 {dest}（源 {item}）")
        shutil.move(str(item), str(dest))
    nested.rmdir()
    return True


def extract_archive(seven_z_path: str | Path, dest_root: str | Path) -> Path:
    """整包解压到 {dest_root}/{date}/。"""
    seven_z_path = Path(seven_z_path)
    if not seven_z_path.is_file():
        raise FileNotFoundError(seven_z_path)
    root = Path(dest_root)
    date = seven_z_path.stem
    day_dir = root / date
    if day_dir.is_dir() and any(day_dir.iterdir()):
        if _flatten_nested_date_dir(day_dir, date):
            print(f"  已修复双层目录，跳过解压: {day_dir}", flush=True)
            return day_dir
        print(f"  跳过解压（目录已存在）: {day_dir}", flush=True)
        return day_dir
    root.mkdir(parents=True, exist_ok=True)
    if _archive_has_date_prefix(seven_z_path, date):
        print(f"  整包解压 {seven_z_path.name} → {root}（归档含日期前缀）", flush=True)
        subprocess.check_call(["tar", "-xf", str(seven_z_path), "-C", str(root)])
    else:
        day_dir.mkdir(parents=True, exist_ok=True)
        print(f"  整包解压 {seven_z_path.name} → {day_dir}", flush=True)
        subprocess.check_call(["tar", "-xf", str(seven_z_path), "-C", str(day_dir)])
    _flatten_nested_date_dir(day_dir, date)
    if not day_dir.is_dir() or not any(day_dir.iterdir()):
        raise FileNotFoundError(f"解压后日目录为空: {day_dir}")
    return day_dir
