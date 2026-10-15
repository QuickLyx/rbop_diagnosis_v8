# -*- coding: utf-8 -*-
"""CWRU 四来源排 028 后枚举；枚举/信号键/四折直接复用本地 `dataset.py` 纯函数。

计划 §11：v6 纯函数已本地复制到根 `dataset.py`，本模块 `import dataset` 复用同一份
实现，避免同一 v6 逻辑出现两份副本（缺通道、键不唯一、非有限输入均由其报错）。
本模块只新增计划 §2.1 需要的最小函数：RPM 读取（特殊键规则 + 载荷档回退）与
来源几何选择；几何只依赖 `source` 与测点，不接收诊断标签，改标签不改提取公式。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import dataset as _v6ds  # noqa: E402  v6 本地副本（枚举/信号键/四折的唯一定义）

from bearing_facts import CWRU_DE_ORDERS, CWRU_FE_ORDERS  # noqa: E402

#: 四来源目录名（计划 §2）：153 文件单元全部取自这四个目录，DE/FE 各一条记录。
#: `_v6ds.SOURCE_DATASETS` 只是 v6 的历史默认值，v8 枚举与四折都不再用它限制来源。
SOURCES = ("Normal", "12k_DE", "12k_FE", "48k_DE")
#: 单传感器通道；DE/FE 作独立单通道样本（沿用 `_v6ds.ALL_CHANNELS`）。
ALL_CHANNELS = _v6ds.ALL_CHANNELS
#: 只读数据根：v6 仓库自带 Data（计划 §2 冻结；调用方可用 config 显式覆盖）。
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "rbop_diagnosis_v6" / "Data"
#: RPM 确实缺失时按文件名末尾载荷档回退（计划 §2.1 冻结），逐条记录回退。
RPM_FALLBACK = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1720.0}


def default_data_root():
    """返回冻结数据根；调用方仍可显式传入 data_root 覆盖。"""
    return DEFAULT_DATA_ROOT


def load_digit(filename):
    """文件名末尾载荷档数字（0–3）；CWRU 文件命名固定以 `_载荷` 结尾。"""
    import re
    match = re.search(r"_(\d+)$", Path(str(filename)).stem)
    if not match:
        raise ValueError(f"无法从文件名解析载荷档：{filename}")
    digit = int(match.group(1))
    if digit not in RPM_FALLBACK:
        raise ValueError(f"载荷档不在冻结回退表内：{filename} -> {digit}")
    return digit


def read_rpm(path, source, filename, channel):
    """按计划 §2.1 读取该记录 RPM；只返回数值，来源标记见 `read_rpm_detail`。"""
    return read_rpm_detail(path, source, filename, channel)[0]


def read_rpm_detail(path, source, filename, channel):
    """读取该记录 RPM 并标注来源：`field` 读自该前缀字段，`fallback` 按载荷档回退。

    先按 v6 特殊键规则选信号前缀，再匹配该前缀的 RPM 字段；匹配不唯一、复数、
    非有限或非正均报错。确实缺失时按文件名载荷档回退，不借用其他前缀 RPM。
    """
    raw, keys = _v6ds._mat_keys(Path(path))
    prefix = _v6ds._signal_prefix(keys, source, filename, channel)
    wanted = (f"{prefix}RPM", f"{prefix}_RPM")
    candidates = [key for key in keys if key in wanted]
    if len(candidates) > 1:
        raise ValueError(f"{source}/{filename} 的 RPM 字段不唯一：{candidates}")
    if not candidates:
        return RPM_FALLBACK[load_digit(filename)], "fallback"
    values = raw[candidates[0]]
    if np.iscomplexobj(values):
        raise ValueError(f"{source}/{filename} 的 RPM 字段不能为复数：{candidates[0]}")
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
        raise ValueError(f"{source}/{filename} 的 RPM 字段不是正标量：{candidates[0]}")
    return float(value[0]), "field"


def fault_orders(source, channel):
    """计划 §2.1 的来源几何：故障按来源端轴承，Normal 按测点对应轴承。

    `12k_DE/48k_DE` 的两传感器都用驱动端（DE）几何；`12k_FE` 的两传感器都用
    风扇端（FE）几何；Normal 无故障来源，DE 测点配 DE 几何、FE 测点配 FE 几何。
    只接收 source 与测点，不读取任何诊断标签。返回冻结阶次字典。
    """
    if channel not in _v6ds.ALL_CHANNELS:
        raise ValueError(f"未知通道：{channel}")
    if source == "Normal":
        return CWRU_DE_ORDERS if channel == "DE" else CWRU_FE_ORDERS
    if source in ("12k_DE", "48k_DE"):
        return CWRU_DE_ORDERS
    if source == "12k_FE":
        return CWRU_FE_ORDERS
    raise ValueError(f"未知 CWRU 来源：{source}")


def enumerate_recordings(data_root=None):
    """153 个文件各产生 DE、FE 两条记录；文件是划分单元（计划 §2 冻结口径）。"""
    from datasets.recording import Recording
    files = _v6ds.enumerate_files(data_root or DEFAULT_DATA_ROOT, SOURCES, exclude_028=True)
    return [Recording(
        record_key=f"{r['file_key']}::{channel}", unit_key=r["file_key"],
        path=Path(r["path"]), dataset="cwru", channel=channel,
        label=r["label"], src_fs=r["raw_fs"], source=r["dataset"],
    ) for r in files for channel in ALL_CHANNELS]


def read_signal(record):
    """返回原采样率向量；`record` 为 Recording，特殊键前缀与唯一性检查沿用 v6。"""
    if record.dataset != "cwru" or record.source not in SOURCES:
        raise ValueError(f"不是有效 CWRU 记录：{record.record_key}")
    return _v6ds.read_channel_signal({
        "path": str(record.path), "dataset": record.source,
        "filename": record.path.name, "file_key": record.unit_key,
    }, record.channel)


def split_source_units(records, split_seed=0):
    """按类别单元级四折划分；随机数消耗、排序与 v6 `split_source_files` 保持一致。

    `records` 为 Recording 或同字段的 SimpleNamespace（缓存记录行）。
    """
    by_unit, seen = {}, set()
    for record in records:
        if record.dataset != "cwru" or record.label not in range(4):
            raise ValueError(f"源四折只能使用有四类真值的 CWRU 记录：{record.record_key}")
        if record.record_key in seen:
            raise ValueError(f"记录键重复：{record.record_key}")
        seen.add(record.record_key)
        if record.unit_key in by_unit and by_unit[record.unit_key] != record.label:
            raise ValueError(f"同单元标签不一致：{record.unit_key}")
        by_unit[record.unit_key] = record.label
    if not by_unit:
        raise ValueError("源四折记录为空")
    rng = np.random.default_rng(split_seed)
    parts = {}
    for label in range(4):
        keys = np.asarray(sorted(key for key, value in by_unit.items() if value == label))
        parts[label] = np.array_split(keys[rng.permutation(len(keys))], 4)
    folds = []
    for fold in range(4):
        val = sorted(str(key) for label in parts for key in parts[label][fold])
        train = sorted(set(by_unit) - set(val))
        if set(train) & set(val) or set(train) | set(val) != set(by_unit):
            raise ValueError("源四折划分未完整覆盖单元或产生重叠")
        folds.append({"fold": fold, "train": train, "val": val})
    return folds
