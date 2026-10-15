# -*- coding: utf-8 -*-
"""v6 数据入口：文件级枚举、通道键规则、48k 降采样、512 点切窗、源四折与 v6 目标划分。

由 v5 `dataset.py` 迁移而来（见 `docs/v5复用与迁移清单.md` §1–§2），做减法：
删除 `TARGET_QUOTAS`/`SOURCE_INDEX`/`ARM_QUOTAS`/`STRATIFIED_OR`/`ARM_NAMES` 与
`split_target_files`（v1 配额档）、`split_target_v2`（v2 元数据分层四档）。
v6 的目标划分不再由随机或元数据配额决定，而是读 `selection.py` 产出的标注池：
`test = 全部目标文件 − 池`，与 `fold`/`seed` 无关（协议 §4 的划分轴塌缩）。

关键不变量（注释即契约）：
- 源共 56 文件：Normal 4 个 + 12k_DE 排除 028 后 52 个；四类 4/12/12/28。
- 源按类别文件级四折划分（`split_seed=0`）：每折 42 训练 / 14 验证。
- 目标 12k_FE 共 45 文件、48k_DE 共 52 文件，均无 Normal，不排 028。
- 同一文件的全部窗口归同一集合；先划分文件再切窗，禁止随机拆窗口。
- 三个特殊文件固定键前缀（X099／X175／X217），其余文件要求候选通道键唯一，否则报错。
- 48k 来源只用 `resample_poly(x, 1, 4)` 降到 12 kHz，再按 512 点不重叠切窗，末尾不足一窗丢弃。
"""
from collections import Counter
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

#: 文件名故障部位前缀到整数标签的映射；分类顺序固定为 Normal、Ball、InnerRace、OuterRace。
LABEL_MAP = {"N": 0, "B": 1, "IR": 2, "OR": 3}
#: 固定类别顺序，必须与模型四分类输出维度一致。
LABEL_NAMES = ["Normal", "Ball", "InnerRace", "OuterRace"]
#: 源训练/验证来源：Normal 4 文件 + 12k_DE 排 028 后 52 文件，共 56 文件。
SOURCE_DATASETS = ("Normal", "12k_DE")
#: 目标适配/测试来源：分别选标、分别评估，不混合两来源的标注池。
TARGET_DATASETS = ("12k_FE", "48k_DE")
#: 单传感器通道；DE/FE 作独立单通道样本，不拼双通道输入。
ALL_CHANNELS = ("DE", "FE")
#: 各来源原始采样率；Normal 与 48k_DE 为 48 kHz，需先降到 12 kHz 再切窗。
RAW_FS = {"Normal": 48000, "12k_DE": 12000, "12k_FE": 12000, "48k_DE": 48000}
#: 三个特殊文件的信号键前缀（不依赖 .mat 键顺序，其余文件要求候选键唯一）。
SPECIAL_PREFIX = {
    ("Normal", "Normal_2.mat"): "X099",
    ("48k_DE", "IR014_1.mat"): "X175",
    ("48k_DE", "IR021_3.mat"): "X217",
}
#: v6 输出根与冻结标注池的默认位置（selection.py 写、target_split_v6 读）。
V6_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "v6_round1"
DEFAULT_SELECTION_PATH = V6_OUTPUT_ROOT / "selection" / "selection.json"


def default_data_root():
    """v8 冻结数据根：`../rbop_diagnosis_v6/Data`（计划 §2；configs/stage3.json 同值）。

    v8 不再回退 v4 目录；其他根由 configs/stage3.json 显式给出。
    """
    return Path(__file__).resolve().parent.parent / "rbop_diagnosis_v6" / "Data"


def parse_label(filename):
    """按文件名起始前缀解析四类标签（N/B/IR/OR），只取 `Path.stem`。"""
    stem = Path(str(filename)).stem
    if re.match(r"^Normal(?:_|$)", stem):
        return "N"
    if re.match(r"^IR\d", stem):
        return "IR"
    if re.match(r"^OR\d", stem):
        return "OR"
    if re.match(r"^B\d", stem):
        return "B"
    raise ValueError(f"无法识别文件标签：{filename}")


def fault_group(filename):
    """返回故障文件的 (标签, 尺寸, 位置)；B/IR 位置记 0，如 OR007@12_0 → (3, 7, 12)。

    只服务诊断落盘与事后分析；选标路径（`selection.py`）与训练路径都不读它。
    """
    match = re.match(r"^(B|IR|OR)(\d{3})(?:@(\d+))?_", Path(str(filename)).stem + "_")
    if not match:
        raise ValueError(f"无法解析故障尺寸/位置：{filename}")
    return LABEL_MAP[match.group(1)], int(match.group(2)), int(match.group(3) or 0)


def is_028_file(filename):
    """判断文件是否为源侧排除的 028 档（源用 True，目标保留）。"""
    return bool(re.search(r"(?:B|IR|OR)028(?:@|_|$)", Path(str(filename)).stem))


def file_key(dataset, filename):
    """生成携带来源目录的稳定文件键，如 `12k_DE/B007_0.mat`。"""
    return f"{dataset}/{Path(str(filename)).name}"


def _resolve_root(data_root):
    """把数据根规整到包含各来源子目录的那一层；误传来源目录时回退一级。"""
    root = Path(data_root)
    if root.name in RAW_FS:
        root = root.parent
    return root


def _mat_keys(path):
    """读取 `.mat` 并返回（原始字典，有效键列表）；有效键指非 `__` 开头的字符串键。"""
    raw = loadmat(path)
    return raw, [key for key in raw if isinstance(key, str) and not key.startswith("__")]


def _signal_prefix(raw_keys, dataset, filename, channel):
    """按固定规则选择该通道的信号前缀；缺键或候选不唯一即报错。

    三个特殊文件直接取固定前缀，其余文件要求 `_{channel}_time` 后缀的候选键
    恰好唯一，防止换数据打包顺序后静默换通道。RPM 字段读取复用同一前缀规则。
    """
    special = SPECIAL_PREFIX.get((dataset, filename))
    suffix = f"_{channel}_time"
    if special:
        wanted = f"{special}{suffix}"
        if wanted not in raw_keys:
            raise ValueError(f"特殊文件 {dataset}/{filename} 缺少固定键 {wanted}")
        return special
    candidates = [key for key in raw_keys if key.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError(
            f"文件 {dataset}/{filename} 的 {channel} 候选键数量为 {len(candidates)}，无法安全选择"
        )
    return candidates[0][: -len(suffix)]


def _channel_key(raw_keys, dataset, filename, channel):
    """由前缀还原完整信号键（保持 v6 行为，供 read_channel_signal 使用）。"""
    prefix = _signal_prefix(raw_keys, dataset, filename, channel)
    return f"{prefix}_{channel}_time"


def read_channel_signal(record, channel):
    """读取一个文件的一个通道，返回 `(N,)` 有限浮点数组。

    未知通道、非一维输入、空信号、非有限值一律报错；不复制通道、不补零伪装另一传感器。
    """
    if channel not in ALL_CHANNELS:
        raise ValueError(f"未知通道：{channel}")
    raw, keys = _mat_keys(Path(record["path"]))
    key = _channel_key(keys, record["dataset"], record["filename"], channel)
    values = np.asarray(raw[key])
    if values.ndim > 2 or (values.ndim == 2 and min(values.shape) > 1):
        raise ValueError(f"通道输入必须是一维向量：{record['file_key']}::{channel}，形状为 {values.shape}")
    signal = values.reshape(-1).astype(np.float64, copy=False)
    if signal.size == 0:
        raise ValueError(f"信号为空：{record['file_key']}::{channel}")
    if not np.isfinite(signal).all():
        raise ValueError(f"信号包含非有限值：{record['file_key']}::{channel}")
    return signal


def read_record_signal(record, channel, target_fs=12000):
    """读取并统一到 12 kHz；48 kHz 只允许 `resample_poly(x, 1, 4)`。

    降采样发生在切窗之前，保证每个 512 点窗口代表相同的物理时间尺度（约 42.67 ms）。
    """
    signal = read_channel_signal(record, channel)
    raw_fs = int(record["raw_fs"])
    if raw_fs != target_fs:
        if raw_fs != 48000 or target_fs != 12000:
            raise ValueError(f"不支持的采样率转换：{raw_fs} -> {target_fs}")
        signal = resample_poly(signal, 1, 4)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 512:
        raise ValueError(f"信号不足一个窗口：{record['file_key']}::{channel}")
    return signal


def window_count(record, channel="DE"):
    """返回一个文件在 12 kHz / 512 点口径下的完整窗口数（末尾不足一窗丢弃）。

    selection.py 的短记录过滤用它；只读信号长度，不构建特征。
    """
    return int(len(read_record_signal(record, channel)) // 512)


# ---------------------------------------------------------------------------
# 文件枚举：先定文件身份，再切窗
# ---------------------------------------------------------------------------
def enumerate_files(data_root=None, datasets=SOURCE_DATASETS, exclude_028=False):
    """枚举指定来源，返回文件记录列表；文件键含来源目录。"""
    root = _resolve_root(data_root or default_data_root())
    records = []
    for dataset in datasets:
        folder = root / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"缺少数据目录：{folder}")
        for path in sorted(folder.glob("*.mat")):
            if exclude_028 and is_028_file(path.name):
                continue
            letter = parse_label(path.name)
            records.append({
                "file_key": file_key(dataset, path.name),
                "dataset": dataset,
                "filename": path.name,
                "path": str(path),
                "label_letter": letter,
                "label": LABEL_MAP[letter],
                "raw_fs": RAW_FS[dataset],
            })
    if not records:
        raise ValueError(f"没有找到数据文件：{root}")
    return records


def enumerate_source_files(data_root=None):
    """枚举源 56 文件：Normal 4 个 + 12k_DE 排 028 后 52 个（四类 4/12/12/28）。"""
    return enumerate_files(data_root, SOURCE_DATASETS, exclude_028=True)


def enumerate_target_files(data_root=None, target_source=None):
    """枚举目标文件：默认 12k_FE 45 个 + 48k_DE 52 个（均无 Normal，不排 028）。"""
    datasets = TARGET_DATASETS if target_source is None else (target_source,)
    if target_source is not None and target_source not in TARGET_DATASETS:
        raise ValueError(f"未知目标来源：{target_source}")
    return enumerate_files(data_root, datasets, exclude_028=False)


# ---------------------------------------------------------------------------
# 分段集合：同一文件的窗口占一段连续行
# ---------------------------------------------------------------------------
class SegmentSet:
    """按文件连续存储的窗口集合；每个文件占一段连续行区间，行号即身份。"""

    def __init__(self, file_keys, signals, labels, segment_ids, fs=12000):
        self.file_keys = [str(value) for value in file_keys]
        self.signals = np.asarray(signals, dtype=np.float64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.segment_ids = np.asarray(segment_ids, dtype=np.int64)
        self.fs = int(fs)
        n = len(self.labels)
        if len(self.file_keys) != n or len(self.segment_ids) != n or len(self.signals) != n:
            raise ValueError("SegmentSet 字段长度不一致")
        self._spans = {}
        for index, key in enumerate(self.file_keys):
            if key in self._spans:
                self._spans[key][1] = index + 1
            else:
                self._spans[key] = [index, index + 1]

    def __len__(self):
        return len(self.labels)

    def span(self, key):
        """返回单个文件的连续行区间 `slice`；键缺失时报错。"""
        if str(key) not in self._spans:
            raise KeyError(f"不存在的文件键：{key}")
        start, stop = self._spans[str(key)]
        return slice(start, stop)

    def rows(self, keys):
        """返回给定文件键的全部行号，按给定文件顺序和文件内段序排列。"""
        rows = []
        for key in keys:
            span = self.span(key)
            rows.extend(range(span.start, span.stop))
        return np.asarray(rows, dtype=np.int64)


def build_segments(records, file_keys, channel, segment_length=512):
    """先按文件读取和降采样，再做不重叠切窗；同一文件的全部窗口连续存放。"""
    if segment_length <= 0:
        raise ValueError("segment_length 必须为正数")
    by_key = {record["file_key"]: record for record in records}
    keys = list(file_keys)
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        raise KeyError(f"未知文件键：{unknown[:3]}")
    row_keys, signals, labels, segment_ids = [], [], [], []
    for key in keys:
        record = by_key[key]
        signal = read_record_signal(record, channel)
        n_windows = len(signal) // segment_length
        if n_windows < 1:
            raise ValueError(f"文件不足一个窗口：{key}::{channel}")
        for segment_id in range(n_windows):
            start = segment_id * segment_length
            row_keys.append(key)
            signals.append(signal[start:start + segment_length])
            labels.append(record["label"])
            segment_ids.append(segment_id)
    return SegmentSet(row_keys, np.stack(signals), labels, segment_ids)


# ---------------------------------------------------------------------------
# 文件级划分：源四折照用 v5；目标读冻结池
# ---------------------------------------------------------------------------
def _records_by_label(records):
    """按整数标签把文件记录分组，供划分与计数复用。"""
    groups = {label: [] for label in range(len(LABEL_NAMES))}
    for record in records:
        groups[int(record["label"])].append(record)
    return groups


def split_source_files(records, split_seed=0, n_folds=4):
    """按类别文件级四折划分，返回每折 train/val 文件键（每折 42/14）。"""
    if n_folds != 4:
        raise ValueError("v6 固定使用四折")
    groups = _records_by_label(records)
    rng = np.random.default_rng(split_seed)
    fold_parts = {label: [] for label in groups}
    for label in sorted(groups):
        keys = np.asarray(sorted(record["file_key"] for record in groups[label]))
        fold_parts[label] = [list(part) for part in np.array_split(keys[rng.permutation(len(keys))], n_folds)]
    folds = []
    all_keys = {record["file_key"] for record in records}
    for fold in range(n_folds):
        val = sorted(key for label in fold_parts for key in fold_parts[label][fold])
        train = sorted(all_keys - set(val))
        folds.append({"fold": fold, "train": train, "val": val})
    return folds


def _class_counts(keys, by_key):
    """统计给定文件键在四个类别上的数量（含零类）。"""
    counts = Counter(int(by_key[key]["label"]) for key in keys)
    return {label: int(counts.get(label, 0)) for label in range(len(LABEL_NAMES))}


def target_split_v6(data_root=None, selection_path=None):
    """读冻结标注池，返回 `{source: {pool/test/all/...}}`；`test = 全部 − 池`。

    三条检查是数据完整性，不是防御（见 `../agents.md`）：
    1. 池与测试不得重叠，且并集恰为该来源全部目标文件；
    2. 池内文件必须属于该来源且真实存在；
    3. 每个 `类×尺寸×位置` 层的测试文件数写进返回值（空层必须可见，不静默通过）。
    池与 `fold`/`seed` 无关，所以这里没有划分种子参数（协议 §4 的塌缩）。
    """
    import json

    path = Path(selection_path or DEFAULT_SELECTION_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结标注池：{path}（先运行 selection.py）")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = enumerate_target_files(data_root)
    by_key = {record["file_key"]: record for record in records}
    result = {}
    for dataset in TARGET_DATASETS:
        keys_all = sorted(record["file_key"] for record in records if record["dataset"] == dataset)
        if dataset not in payload["sources"]:
            raise ValueError(f"selection.json 缺少来源 {dataset}")
        pool = sorted(str(key) for key in payload["sources"][dataset]["pool"])
        unknown = [key for key in pool if key not in by_key]
        if unknown:
            raise ValueError(f"{dataset} 池内存在未枚举的目标文件：{unknown[:3]}")
        if any(by_key[key]["dataset"] != dataset for key in pool):
            raise ValueError(f"{dataset} 池内混入其他来源文件")
        test = sorted(set(keys_all) - set(pool))
        if set(pool) & set(test):
            raise ValueError(f"{dataset} 池与测试列表重叠")
        if set(pool) | set(test) != set(keys_all):
            raise ValueError(f"{dataset} 池与测试并集不等于全部目标文件")
        groups = {}
        for key in keys_all:
            groups.setdefault(fault_group(key), []).append(key)
        result[dataset] = {
            "all": keys_all,
            "pool": pool,
            "test": test,
            "pool_class_counts": _class_counts(pool, by_key),
            "test_class_counts": _class_counts(test, by_key),
            "test_records": 2 * len(test),
            # 层键形如 "3/7/12"（标签/尺寸/位置）；值为该层测试文件数，0 即空层。
            "test_group_counts": {f"{l}/{s}/{p}": sum(key in test for key in keys)
                                  for (l, s, p), keys in sorted(groups.items())},
            "empty_groups": sorted(f"{l}/{s}/{p}" for (l, s, p), keys in sorted(groups.items())
                                   if not (set(keys) & set(test))),
        }
    return result


def source_fold_counts(folds, records):
    """统计各折训练/验证文件在四类上的数量，供检查与报告复用。"""
    by_key = {record["file_key"]: record for record in records}
    return [{"train": _class_counts(fold["train"], by_key), "val": _class_counts(fold["val"], by_key)}
            for fold in folds]


# ---------------------------------------------------------------------------
# 自检：锁死 v6 文件级不变量（手动跑，不作启动钩子）
# ---------------------------------------------------------------------------
def _self_check():
    """验证源 56 文件与四折 42/14、目标 45/52 文件，以及冻结池划分的完整性。"""
    records = enumerate_source_files()
    assert len(records) == 56, len(records)
    folds = split_source_files(records, 0)
    assert all(len(fold["train"]) == 42 and len(fold["val"]) == 14 for fold in folds)
    assert all(set(fold["train"]).isdisjoint(fold["val"]) for fold in folds)
    assert sum(len(fold["val"]) for fold in folds) == 56
    targets = enumerate_target_files()
    counts = Counter(record["dataset"] for record in targets)
    assert counts["12k_FE"] == 45 and counts["48k_DE"] == 52, dict(counts)
    path = Path(DEFAULT_SELECTION_PATH)
    if path.is_file():
        splits = target_split_v6()
        assert len(splits["12k_FE"]["pool"]) + len(splits["12k_FE"]["test"]) == 45
        assert len(splits["48k_DE"]["pool"]) + len(splits["48k_DE"]["test"]) == 52
        assert not splits["12k_FE"]["empty_groups"], splits["12k_FE"]["empty_groups"]
        assert not splits["48k_DE"]["empty_groups"], splits["48k_DE"]["empty_groups"]
        print("v6 target split:", {src: (len(v["pool"]), len(v["test"])) for src, v in splits.items()})
    else:
        print(f"跳过 target_split_v6 检查（{path} 尚未生成）")
    print("ALL DATASET CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
