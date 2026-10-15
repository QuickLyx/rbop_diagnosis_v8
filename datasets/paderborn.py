# -*- coding: utf-8 -*-
"""PU 记录枚举与原始通道读取，显式落盘官方截断文件。"""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bearing_facts import BY_ID, CONDITIONS, TRUNCATED_RECORDINGS
from .recording import Recording, _as_signal

ROOT = Path(__file__).resolve().parents[1]
#: v8 冻结数据根：v7 的 PU 数据只读（计划 §2；configs/stage3.json data_roots.paderborn 同值）。
DEFAULT_DATA_ROOT = ROOT.parent / "rbop_diagnosis_v7" / "Data" / "paderborn" / "extracted"
DEFAULT_DROPPED_PATH = ROOT / "outputs" / "data" / "dropped_recordings.json"


def enumerate_recordings(data_root=None, include_kb=False, dropped_path=None):
    """按轴承、文件名排序枚举；KA08 坏文件整条丢弃，KB 默认排除。"""
    root = Path(data_root or DEFAULT_DATA_ROOT)
    records, dropped = [], []
    for bearing_id in sorted(BY_ID):
        bearing = BY_ID[bearing_id]
        if bearing.is_kb and not include_kb:
            continue
        folder = root / bearing_id
        if not folder.is_dir():
            raise FileNotFoundError(f"缺少轴承目录：{folder}")
        paths = sorted(folder.glob("*.mat"))
        expected = {f"{c}_{bearing_id}_{rep}.mat" for c in CONDITIONS for rep in range(1, 21)}
        if {p.name for p in paths} != expected:
            raise ValueError(f"轴承记录不完整或含未知记录：{bearing_id}")
        for path in paths:
            condition, file_bearing, rep = path.stem.rsplit("_", 2)
            if condition not in CONDITIONS or file_bearing != bearing_id or not 1 <= int(rep) <= 20:
                raise ValueError(f"无法解析 PU 记录身份：{path}")
            key = f"{bearing_id}/{path.name}::vibration_1"
            if path.stem in TRUNCATED_RECORDINGS:
                dropped.append({
                    "record_key": key, "unit_key": f"{bearing_id}/{condition}",
                    "path": str(path), "filename": path.name,
                    "reason": "官方截断并补零，scipy.loadmat 无法加载；按 D12 整条丢弃",
                })
                continue
            records.append(Recording(
                record_key=key, unit_key=f"{bearing_id}/{condition}", path=path,
                dataset="paderborn", channel="vibration_1", label=bearing.label,
                src_fs=64000, source="paderborn", bearing_id=bearing_id,
                condition=condition, origin=bearing.origin, severity=bearing.severity,
            ))
    path = Path(dropped_path or DEFAULT_DROPPED_PATH)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != dropped:
            raise FileExistsError(f"丢弃记录已有不同内容，不覆盖：{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(dropped, handle, ensure_ascii=False, indent=2)
    return records


def read_channels(path):
    """公开全部原始通道；顶层键、通道名必须唯一且包含 vibration_1。"""
    path = Path(path)
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [key for key in raw if not key.startswith("__")]
    if keys != [path.stem]:
        raise ValueError(f"PU 顶层键必须唯一且等于文件名词干：{path}")
    root = raw[path.stem]
    if not hasattr(root, "Y"):
        raise ValueError(f"PU 文件缺少 Y 通道列表：{path}")
    result = {}
    for channel in np.asarray(root.Y, dtype=object).reshape(-1):
        if not hasattr(channel, "Name") or not hasattr(channel, "Data"):
            raise ValueError(f"PU 通道缺少 Name 或 Data：{path}")
        name = channel.Name
        if not isinstance(name, str) or not name or name in result:
            raise ValueError(f"PU 通道名无效或重复：{path}，{name}")
        result[name] = _as_signal(channel.Data)
    if "vibration_1" not in result:
        raise ValueError(f"PU 文件缺少 vibration_1：{path}")
    return result


def read_signal(record):
    """公开单条振动记录的原采样率向量。"""
    if record.dataset != "paderborn" or record.channel != "vibration_1":
        raise ValueError(f"不是有效 PU 振动记录：{record.record_key}")
    return read_channels(record.path)[record.channel]
