# -*- coding: utf-8 -*-
"""阶段三固定采样、双通道损失、AdaBN 与训练循环；调用训练函数须另获执行授权。

只服务 configs/stage3.json 冻结矩阵（阶段三执行计划 §5/§10）：
- 九个深度臂固定 `[0,1,2,3]×[0,1,2]`、batch=64、epochs=30、Adam 默认 β/ε、
  无 weight decay/调度；从头臂 lr=joint_lr，ce_* 臂 lr=adapt_lr；
- 从头臂源 sampler 独立使用相同 seed，目标 seed+1；Mixup 独立 RNG=seed+10000；
- 双通道输入逐 batch 组装：第一通道 logspec，第二通道前 25 维缩放后物理摘要、
  后 231 位补零，不永久缓存补零通道；
- ce_* 臂用显式池名映射，不做字符串前缀解析；
- AdaBN 无优化器、Dropout 关闭、只允许 BN 缓冲变化；
- 正式每轮写 last.pt、结束后落 best.pt 作诊断；测试前向 eval/no_grad 不改 BN。
"""
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from evaluate import aggregate_records, aggregate_units
from features import scale_phys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: 池名到目标监督池的显式映射（阶段三 §5：禁止把 ce_K19_gg 解析成预算名）。
POOL_ARM_MAP = {"ce_K19": "K19_old", "ce_K19_gg": "K19_gg", "ce_K116": "K116",
                "ce_K19_logspec": "K19_old"}

#: v8 九个深度运行臂（96 梯度格 + 12 AdaBN 格的全部训练入口）。
DEEP_ARMS = ("joint", "joint_logspec", "mixup4src", "coral4src", "adabn",
             "ce_K19", "ce_K19_gg", "ce_K116", "ce_K19_logspec")

#: 从头训练臂（lr=joint_lr）；ce_* 臂用 adapt_lr。
SCRATCH_ARMS = ("joint", "joint_logspec", "mixup4src", "coral4src")

#: 类条件 CORAL 只在三个故障来源间对齐 B/IR/OR；Normal 只走源 CE（§5.2）。
CORAL_CLASSES = (1, 2, 3)

#: CWRU 来源名到 Mixup/CORAL 组名的固定映射（四来源分组身份）。
SOURCE_GROUP = {"Normal": "Normal", "12k_DE": "12k_DE", "12k_FE": "12k_FE", "48k_DE": "48k_DE"}


def seed_everything(seed):
    """沿用 v6/v7 的随机性设置。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_window_weights(feature):
    """照搬 v6 四步均衡：通道、类别、记录、记录内窗口。"""
    if len(feature.y) == 0:
        raise ValueError("空特征集不能计算采样权重")
    keys = np.asarray(feature.record_keys, dtype=object)
    channels = np.asarray(feature.channels, dtype=object)
    labels = feature.y.detach().cpu().numpy()
    if np.any(labels < 0):
        raise ValueError("无标签目标不能计算真值均衡权重")
    record_pairs = {(str(channel), str(key), int(label))
                    for channel, key, label in zip(channels, keys, labels)}
    record_counts, window_counts = {}, {}
    for channel, key, label in record_pairs:
        group = (str(channel), int(label))
        record_counts[group] = record_counts.get(group, 0) + 1
        record_key = (str(channel), str(key))
        window_counts[record_key] = int(np.sum((channels == channel) & (keys == key)))
    weights = np.empty(len(labels), dtype=np.float64)
    for index, (channel, key, label) in enumerate(zip(channels, keys, labels)):
        group = (str(channel), int(label))
        record_key = (str(channel), str(key))
        weights[index] = 1.0 / (record_counts[group] * window_counts[record_key])
    return weights


def make_weighted_loader(feature, batch_size, num_samples, seed,
                         class_balance=True, labeled=True):
    """固定样本数、有放回；第三个元素是行号，供逐 batch 组装 phys25/来源组。

    关闭均衡或无标签时只使用均匀窗口权重；无标签分支不访问 feature.y。
    """
    if len(feature.X) == 0 or batch_size <= 0 or num_samples < batch_size:
        raise ValueError("特征不能为空，采样数须不少于正数 batch_size")
    weights = torch.as_tensor(balanced_window_weights(feature), dtype=torch.double) \
        if labeled and class_balance else torch.ones(len(feature.X), dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, num_samples=int(num_samples), replacement=True,
        generator=torch.Generator().manual_seed(int(seed)))
    # 无标签分支不访问 feature.y，防止采样器接触目标真值。
    labels = feature.y if labeled else torch.full((len(feature.X),), -1, dtype=torch.long)
    row_ids = torch.arange(len(feature.X), dtype=torch.long)
    loader = DataLoader(TensorDataset(feature.X, labels, row_ids), batch_size=batch_size,
                        sampler=sampler, drop_last=True)
    return loader, weights


def steps_per_epoch(s_ref, batch_size):
    """只以本折源训练 DE 窗口数确定所有臂的固定预算。"""
    if s_ref <= 0 or batch_size <= 0:
        raise ValueError("s_ref 和 batch_size 必须为正数")
    return int(math.ceil(s_ref / batch_size))


def row_sources(feature):
    """逐窗口的来源名数组（长度=窗口数），供 Mixup 分组与 CORAL。"""
    rows = []
    for record in feature.records:
        rows.extend([str(record["source"])] * (record["stop"] - record["start"]))
    if len(rows) != len(feature.X):
        raise ValueError("来源数组与窗口数不一致")
    return np.asarray(rows, dtype=object)


# ---------------------------------------------------------------------------
# 双通道输入组装：训练逐 batch、验证逐批切片，均不永久缓存补零通道
# ---------------------------------------------------------------------------
def prepare_batch(xb, phys_rows, scaler, logspec_only, device):
    """组装一个 batch 的 (B,2,256)：第二通道先缩放物理摘要再补零。

    logspec_only=True 时第二通道恒零（输入消融臂），但 scaler 仍必须提供，
    保证消融臂走完全相同的组装代码路径。
    """
    if scaler is None or set(scaler) != {"mean", "std"} or len(scaler["mean"]) != 25 \
            or len(scaler["std"]) != 25:
        raise ValueError("双通道组装需要源训练 scaler（mean/std 各 25 维）")
    second = torch.zeros((xb.size(0), 256), dtype=xb.dtype, device=device)
    if not logspec_only:
        if phys_rows is None or len(np.asarray(phys_rows)) != xb.size(0):
            raise ValueError("物理摘要行数与 batch 不一致")
        scaled = torch.from_numpy(scale_phys(np.asarray(phys_rows), scaler)).to(device)
        second[:, :25] = scaled.to(xb.dtype)
    # 第一通道取 logspec 的 (B,256) 视图，与补零后的第二通道沿通道维拼接。
    return torch.stack((xb[:, 0].to(device), second), dim=1)


# ---------------------------------------------------------------------------
# 四来源 Mixup（§5.1）与软 CE
# ---------------------------------------------------------------------------
def _soft_cross_entropy(logits, soft_labels):
    """软标签 CE：mean(-sum(y' * log_softmax(logits)))；y' 各行必须和为 1。"""
    if not torch.allclose(soft_labels.sum(dim=1),
                          torch.ones(len(soft_labels), device=soft_labels.device), atol=1e-5):
        raise ValueError("Mixup 软标签各行和必须为 1")
    return -(soft_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _dirichlet_uniform(rng, k=4):
    """Dirichlet(1,...,1) 精确采样：k 个 Exp(1) 归一化；只用传入的独立 RNG。"""
    weights = torch.empty(k).exponential_(1.0, generator=rng)
    total = weights.sum()
    if total <= 0:  # 概率为零的退化保护
        return torch.full((k,), 1.0 / k)
    return weights / total


def mixup_source_batch(inputs_source, labels_source, embeddings, group_keys, mixup_rng):
    """按来源分组组成 M 个四元组，返回混合输入、混合特征与共享软标签。

    - M=min(各组数量)；M=0 时返回 None（两项混合损失为零并计次数）；
    - 各组独立随机排列后取前 M 个；每组 lambda~Dirichlet(1,1,1,1)；
    - x_mix 由调用方再过骨干，h_mix 由调用方直接走原 head；
    - 只消耗 mixup_rng，绝不触碰基座 sampler 的生成器。
    """
    group_ids = {}
    for index, key in enumerate(group_keys):
        group_ids.setdefault(str(key), []).append(index)
    if not group_ids:
        return None
    m = min(len(ids) for ids in group_ids.values())
    if m == 0:
        return None
    ordered = sorted(group_ids)
    picks = []
    for key in ordered:
        ids = group_ids[key]
        perm = torch.randperm(len(ids), generator=mixup_rng)
        picks.append([ids[j] for j in perm[:m].tolist()])
    x_mix, h_mix, soft = [], [], []
    for column in range(m):
        lam = _dirichlet_uniform(mixup_rng, len(ordered)).to(inputs_source.device)
        rows = [picks[g][column] for g in range(len(ordered))]
        x_mix.append(sum(lam[g] * inputs_source[rows[g]] for g in range(len(rows))))
        h_mix.append(sum(lam[g] * embeddings[rows[g]] for g in range(len(rows))))
        onehot = F.one_hot(labels_source[rows], num_classes=4).float()
        soft.append((lam[:len(rows), None] * onehot).sum(dim=0))
    return torch.stack(x_mix), torch.stack(h_mix), torch.stack(soft)


# ---------------------------------------------------------------------------
# 类条件 CORAL（§5.2）
# ---------------------------------------------------------------------------
def _covariance(embeddings):
    """n-1 分母的样本协方差。"""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (embeddings.size(0) - 1)


def class_conditional_coral(embeddings_source, labels_source, group_keys,
                            classes=CORAL_CLASSES):
    """只在故障来源间按共有 B/IR/OR 分别对齐 256 维嵌入。

    损失 = ||C_a-C_b||_F^2/(4*256^2)，有效来源对×类别等权平均；
    任一侧不足两样本的项跳过并记覆盖；全部无效返回设备上的标量零。
    """
    source_groups = {}
    for index, key in enumerate(group_keys):
        source_groups.setdefault(str(key), []).append(index)
    if len(source_groups) < 2:
        return embeddings_source.new_zeros(()), []
    names = sorted(source_groups)
    terms, coverage = [], []
    dimension = embeddings_source.size(1)
    for label in classes:
        masks = {name: labels_source[source_groups[name]] == label for name in names}
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                left, right = masks[names[first]], masks[names[second]]
                pair = f"{names[first]}|{names[second]}"
                if int(left.sum()) < 2 or int(right.sum()) < 2:
                    coverage.append({"class": int(label), "pair": pair, "skipped": True})
                    continue
                covariance_left = _covariance(embeddings_source[source_groups[names[first]]][left])
                covariance_right = _covariance(embeddings_source[source_groups[names[second]]][right])
                terms.append(((covariance_left - covariance_right) ** 2).sum() / (4 * dimension ** 2))
                coverage.append({"class": int(label), "pair": pair, "skipped": False})
    if not terms:
        return embeddings_source.new_zeros(()), coverage
    align = torch.stack(terms).mean()
    if not torch.isfinite(align):
        raise FloatingPointError("CORAL 对齐项出现非有限值")
    return align, coverage


def compute_losses(arm, logits_source, labels_source, logits_target=None, labels_target=None,
                   align=None, ce_input_mix=None, ce_feature_mix=None):
    """只组装各臂损失；Mixup/CORAL 的混合/对齐项由训练循环预先计算传入。"""
    ce_source = F.cross_entropy(logits_source, labels_source)
    ce_target, align_value = ce_source.new_zeros(()), ce_source.new_zeros(())
    if arm in SCRATCH_ARMS:
        if arm == "mixup4src":
            # 两项混合损失各配 0.5；M=0 时调用方传 None，保持零。
            if ce_input_mix is not None:
                align_value = align_value + 0.5 * ce_input_mix
            if ce_feature_mix is not None:
                align_value = align_value + 0.5 * ce_feature_mix
        elif arm == "coral4src":
            if align is None:
                raise ValueError("coral4src 必须提供类条件 CORAL 对齐项")
            align_value = 0.2 * align
    elif arm in POOL_ARM_MAP:
        if logits_target is None or labels_target is None:
            raise ValueError("ce_* 臂必须提供目标池 batch 的 logits 与标签")
        ce_target = F.cross_entropy(logits_target, labels_target)
    else:
        raise ValueError(f"没有梯度损失的臂：{arm}")
    return {"loss": ce_source + ce_target + align_value, "ce_source": ce_source,
            "ce_target": ce_target, "align": align_value}


# ---------------------------------------------------------------------------
# 源验证：eval/no_grad，逐批组装双通道；记录级聚合后 DE/FE 等权融合
# ---------------------------------------------------------------------------
def _class_metrics(true, pred):
    """源记录级四类指标，宏 F1 只对有真实样本的类别平均。"""
    confusion = np.zeros((4, 4), dtype=np.int64)
    for actual, guessed in zip(true, pred):
        confusion[actual, guessed] += 1
    support = confusion.sum(axis=1)
    f1 = []
    for label in range(4):
        denominator = support[label] + confusion[:, label].sum()
        f1.append(None if support[label] == 0 else
                  float(2 * confusion[label, label] / denominator))
    present = [value for value in f1 if value is not None]
    return {"accuracy": float(np.mean(np.asarray(true) == np.asarray(pred))) if true else 0.0,
            "macro_f1": float(np.mean(present)) if present else None,
            "f1": f1, "support": support.tolist(), "confusion_matrix": confusion.tolist()}


def source_validation(model, feature, batch_size=256, scaler=None, logspec_only=False,
                      inputs=None):
    """仅源验证：逐批组装双通道输入，记录均值后 DE/FE 文件等权融合；不增加前向。

    inputs 非空时直接使用（供合成小模型测试），否则逐批调 prepare_batch 组装。
    """
    if not len(feature.X):
        raise ValueError("源验证特征为空")
    if scaler is None:
        raise ValueError("源验证必须提供源训练 scaler")
    model.eval()
    device = next(model.parameters()).device
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(feature.X), batch_size):
            xb = feature.X[start:start + batch_size]
            if inputs is not None:
                batch_inputs = inputs[start:start + batch_size].to(device)
            else:
                phys = None if logspec_only else feature.phys25[start:start + batch_size]
                batch_inputs = prepare_batch(xb, phys, scaler, logspec_only, device)
            logits = model(batch_inputs)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("源验证输出包含非有限值")
            probabilities.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(probabilities).numpy()
    record_predictions = aggregate_records(feature.records, probs)
    unit_predictions = aggregate_units(record_predictions)
    groups, by_source = {}, {}
    for record in record_predictions:
        actual = int(record["label"])
        guessed = int(np.asarray(record["probabilities"]).argmax())
        channel, source = record["channel"], record["source"]
        groups.setdefault(channel, ([], []))
        by_source.setdefault(source, {}).setdefault(channel, ([], []))
        for true, pred in (groups[channel], by_source[source][channel]):
            true.append(actual)
            pred.append(guessed)
    if not {"DE", "FE"} <= groups.keys():
        raise ValueError("源验证必须同时包含 DE 和 FE")
    metrics = {channel: _class_metrics(*values) for channel, values in groups.items()}
    metrics["by_source"] = {
        source: {channel: _class_metrics(*values) for channel, values in channels.items()}
        for source, channels in by_source.items()}
    metrics["file_fused"] = _class_metrics(
        [row["label"] for row in unit_predictions],
        [int(np.asarray(row["probabilities"]).argmax()) for row in unit_predictions])
    for source, values in metrics["by_source"].items():
        units = [row for row in unit_predictions if row["source"] == source]
        values["file_fused"] = _class_metrics(
            [row["label"] for row in units],
            [int(np.asarray(row["probabilities"]).argmax()) for row in units])
    return metrics


def select_validation_score(metrics):
    return (float(np.mean([metrics[c]["macro_f1"] for c in ("DE", "FE")])),
            float(np.mean([metrics[c]["accuracy"] for c in ("DE", "FE")])))


# ---------------------------------------------------------------------------
# AdaBN：无优化器，Dropout 关闭，只允许 BN 缓冲变化
# ---------------------------------------------------------------------------
def _set_bn_train_only(model, expected_momentum=0.1):
    """仅供获授权后的 AdaBN：Dropout 关闭，只开启 BN 统计更新。"""
    model.eval()
    found = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            found = True
            if module.momentum != expected_momentum:
                raise ValueError(f"AdaBN 沿用 BN momentum={expected_momentum}，不重置统计")
            module.train()
    if not found:
        raise ValueError("模型没有 BN 层")


def _assert_only_bn_buffers_changed(before, model):
    allowed = {f"{name}.{buffer}" for name, module in model.named_modules()
               if isinstance(module, nn.BatchNorm1d)
               for buffer in ("running_mean", "running_var", "num_batches_tracked")}
    for key, value in model.state_dict().items():
        if key not in allowed and not torch.equal(before[key], value.detach().cpu()):
            raise AssertionError(f"AdaBN 改变了非 BN 缓冲区：{key}")


# ---------------------------------------------------------------------------
# checkpoint 与历史
# ---------------------------------------------------------------------------
def save_checkpoint(path, state_dict, metadata):
    """训练目录由入口独占；每轮保存普通权重和元数据，不做断点续训。"""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({**metadata, "state_dict": state_dict}, temporary)
    temporary.replace(path)


def load_checkpoint(path, model=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint 缺少 state_dict")
    if model is not None:
        model.load_state_dict(payload["state_dict"])
    return payload


def write_history(path, history):
    fields = list(dict.fromkeys(key for row in history for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_model(model, source_feature, target_feature, val_feature, arm,
                config, output, s_ref, seed, run_kind="formal", pilot_epochs=None):
    """获授权后才调用：固定步数跑满，各轮只记录源验证，正式返回 last。

    pilot_epochs 只用于 P4 预算试跑（run_kind="pilot"）；正式运行必须用配置轮数，
    pilot 产物的 checkpoint 元数据带 run_kind=pilot，正式评估会拒绝。
    """
    if arm not in DEEP_ARMS:
        raise ValueError(f"{arm} 不是深度运行臂，禁止送入训练循环")
    params = config["parameters"]
    batch_size, epochs = params["batch_size"], params["epochs"]
    if pilot_epochs is not None:
        if run_kind != "pilot":
            raise ValueError("pilot_epochs 只能用于 run_kind=pilot 的预算试跑")
        if not 1 <= pilot_epochs <= epochs:
            raise ValueError(f"pilot_epochs 必须在 1..{epochs}")
        epochs = int(pilot_epochs)
    steps = steps_per_epoch(s_ref, batch_size)
    logspec_only = arm.endswith("_logspec")
    scaler = getattr(source_feature, "scaler", None)
    if scaler is None:
        raise ValueError("源训练特征子集必须携带 fit_scaler 生成的 scaler")
    phys25_source = torch.as_tensor(source_feature.phys25)
    phys25_target = torch.as_tensor(target_feature.phys25) if target_feature is not None else None
    sources_source = row_sources(source_feature)
    source_loader = None
    if arm != "adabn":
        # 所有从头臂的源 sampler 独立使用相同 seed；sampler 序列不受 Mixup 影响。
        source_loader, _ = make_weighted_loader(source_feature, batch_size,
                                                steps * batch_size, seed, class_balance=True)
    target_loader = None
    if target_feature is not None:
        # 监督池按同款有标签均衡；无标签 AdaBN 均匀窗口采样且不接触真值。
        target_loader, _ = make_weighted_loader(
            target_feature, batch_size, steps * batch_size, seed + 1, class_balance=True,
            labeled=arm != "adabn")
    device = next(model.parameters()).device
    # Mixup 独立 RNG=seed+10000，不改基座 sampler 序列（§5）。
    mixup_rng = torch.Generator().manual_seed(seed + 10000) if arm == "mixup4src" else None
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()} \
        if arm == "adabn" else None
    optimizer = None if arm == "adabn" else torch.optim.Adam(
        model.parameters(), lr=params["joint_lr"] if arm in SCRATCH_ARMS else params["adapt_lr"])
    best, last, history = None, None, []
    started = time.perf_counter()
    epoch_seconds = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        totals = {"ce_source": 0.0, "ce_target": 0.0, "align": 0.0,
                  "ce_input_mix": 0.0, "ce_feature_mix": 0.0}
        mixup_batches = mixup_zero_batches = 0
        coral_pairs = coral_skipped = 0
        if arm == "adabn":
            _set_bn_train_only(model)
            with torch.no_grad():
                for xb_t, _, idx_t in target_loader:
                    inputs_t = prepare_batch(xb_t, phys25_target[idx_t], scaler, False, device)
                    model(inputs_t)
            _assert_only_bn_buffers_changed(before, model)
        else:
            model.train()
            target_iterator = iter(target_loader) if target_loader is not None else None
            for xb_s, yb_s, idx_s in source_loader:
                labels_source = yb_s.to(device)
                inputs_s = prepare_batch(xb_s, phys25_source[idx_s], scaler, logspec_only, device)
                logits_s, _, feat_s = model(inputs_s, return_feat=True)
                ce_input_mix = ce_feature_mix = align = None
                logits_t = labels_t = None
                if arm == "mixup4src":
                    group_keys = [SOURCE_GROUP[key] for key in sources_source[np.asarray(idx_s)]]
                    mixed = mixup_source_batch(inputs_s, labels_source, feat_s,
                                               group_keys, mixup_rng)
                    if mixed is not None:
                        x_mix, h_mix, soft_labels = mixed
                        logits_mix_x = model(x_mix)[0]
                        logits_mix_h = model.head(h_mix)
                        ce_input_mix = _soft_cross_entropy(logits_mix_x, soft_labels)
                        ce_feature_mix = _soft_cross_entropy(logits_mix_h, soft_labels)
                        mixup_batches += 1
                    else:
                        # M=0：两项混合损失为零，次数仍需落 history（§5.1）。
                        mixup_zero_batches += 1
                elif arm == "coral4src":
                    group_keys = [SOURCE_GROUP[key] for key in sources_source[np.asarray(idx_s)]]
                    align, coverage = class_conditional_coral(feat_s, labels_source, group_keys)
                    coral_pairs += len(coverage)
                    coral_skipped += sum(int(item["skipped"]) for item in coverage)
                elif target_iterator is not None:
                    xb_t, yb_t, idx_t = next(target_iterator)
                    labels_t = yb_t.to(device)
                    inputs_t = prepare_batch(xb_t, phys25_target[idx_t], scaler, logspec_only, device)
                    logits_t = model(inputs_t)[0]
                losses = compute_losses(arm, logits_s, labels_source, logits_t, labels_t,
                                        align, ce_input_mix, ce_feature_mix)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"第 {epoch} 轮损失非有限，停止更新")
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()
                for key in ("ce_source", "ce_target", "align"):
                    totals[key] += float(losses[key].detach())
                if ce_input_mix is not None:
                    totals["ce_input_mix"] += float(ce_input_mix.detach())
                if ce_feature_mix is not None:
                    totals["ce_feature_mix"] += float(ce_feature_mix.detach())
        metrics = source_validation(model, val_feature, scaler=scaler, logspec_only=logspec_only)
        score, accuracy = select_validation_score(metrics)
        seconds = time.perf_counter() - epoch_start
        epoch_seconds.append(seconds)
        row = {"epoch": epoch, **{key: value / max(steps, 1) for key, value in totals.items()},
               "mixup_batches": mixup_batches, "mixup_zero_batches": mixup_zero_batches,
               "coral_pairs": coral_pairs, "coral_pairs_skipped": coral_skipped,
               "epoch_seconds": seconds, "run_kind": run_kind,
               "val_macro_f1_mean": score, "val_accuracy_mean": accuracy,
               "updates": 0 if arm == "adabn" else steps * epoch,
               "target_forward_batches_cum": steps * epoch if target_loader is not None else 0}
        for channel in ("DE", "FE", "file_fused"):
            row[f"val_macro_f1_{channel}"] = metrics[channel]["macro_f1"]
            row[f"val_accuracy_{channel}"] = metrics[channel]["accuracy"]
        history.append(row)
        state_dict = {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()}
        last = {"epoch": epoch, "score": score, "accuracy": accuracy, "metrics": metrics,
                "state_dict": state_dict}
        if best is None or (score, accuracy) > (best["score"], best["accuracy"]):
            best = deepcopy(last)
        # 每轮覆盖写 last.pt（正式以最后一轮为准）；best 只在结束后落盘一次作诊断。
        metadata = {**config, "run_kind": run_kind, "arm": arm, "fold": config["fold"],
                    "seed": seed, "checkpoint_kind": "last", "epoch": epoch,
                    "checkpoint_epoch": epoch, "last_epoch": epochs, "best_epoch": best["epoch"],
                    "steps_per_epoch": steps, "steps": steps * epoch,
                    "updates": 0 if arm == "adabn" else steps * epoch,
                    "target_forwards": steps * epoch if target_loader is not None else 0,
                    "logspec_only": logspec_only, "scaler": scaler,
                    # 只有 ce_* 是监督池；adabn 的 U 是无标签目标，监督池必须为空。
                    "supervised_pool_units": list(config.get("target_train_units") or [])
                        if arm in POOL_ARM_MAP else [],
                    "validation_metrics": metrics, "validation_score": score}
        save_checkpoint(Path(output) / "last.pt", state_dict, metadata)
        write_history(Path(output) / "history.csv", history)
        print(f"{arm} 第 {epoch}/{epochs} 轮：源验证 F1={score:.4f}，正式权重=last", flush=True)
    model.eval()
    elapsed = time.perf_counter() - started
    metadata["checkpoint_kind"] = "best"
    metadata["epoch"] = metadata["checkpoint_epoch"] = best["epoch"]
    metadata["elapsed_seconds"] = elapsed
    save_checkpoint(Path(output) / "best.pt", best["state_dict"], metadata)
    # 总/轮耗时落 fit_info.json，供 P4 预算汇总（summarize 只读 json）。
    (Path(output) / "fit_info.json").write_text(json.dumps(
        {"arm": arm, "fold": config["fold"], "seed": seed, "run_kind": run_kind,
         "epochs": epochs, "elapsed_seconds": elapsed, "epoch_seconds": epoch_seconds},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model": model, "best": best, "last": last, "history": history,
            "steps_per_epoch": steps, "epochs": epochs, "elapsed_seconds": elapsed,
            "updates": 0 if arm == "adabn" else steps * epochs}
