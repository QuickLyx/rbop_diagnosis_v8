# -*- coding: utf-8 -*-
"""阶段三评估：只读前向、三级等权聚合、scope 配对评分、晚期融合与不确定报告。

只服务 configs/stage3.json 冻结矩阵：深度/浅层/融合三种评估入口，
默认 dry-run 打印计划，显式 --execute 才做实际前向或读模型。
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Normal", "Ball", "IR", "OR"]
UNIT_FIELDS = ("unit_key", "dataset", "label", "source", "bearing_id", "condition",
               "origin", "severity", "src_fs")
FUSION_BASES = {"fusion_joint": "joint", "fusion_ce_K19": "ce_K19"}
#: 计划 §8 冻结的六个候选权重与平票规则；配置偏离即报错。
FROZEN_WEIGHTS = {(1.0, 0.0, 0.0), (0.75, 0.25, 0.0), (0.75, 0.0, 0.25),
                  (0.5, 0.5, 0.0), (0.5, 0.25, 0.25), (0.5, 0.0, 0.5)}
FROZEN_TIE_BREAK = ["larger_deep", "fewer_auxiliary", "larger_normal"]
FROZEN_ABLATIONS = ["without_phys", "without_normal"]
#: 深度评估 info 必须携带的 last 重跑来源字段（计划 §8：CW/PU 必须同一末轮）。
PROVENANCE_KEYS = ("checkpoint", "checkpoint_kind", "epoch", "last_epoch", "repr_config",
                   "scaler", "source_split", "supervised_pool_units")


def _probabilities(values):
    probabilities = np.asarray(values, dtype=np.float64)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 4
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("预测必须是有限、归一化的四类概率")
    return probabilities


def aggregate_records(records, window_probs):
    """同记录窗口概率取均值；区间必须恰好覆盖当前子集的全部窗口。"""
    if not records and len(window_probs) == 0:
        return []
    probabilities = _probabilities(window_probs)
    result, seen, offset = [], set(), 0
    for record in records:
        key = record["record_key"]
        start, stop = record["start"], record["stop"]
        if key in seen:
            raise ValueError(f"记录重复：{key}")
        if (type(start) is not int or type(stop) is not int or start != offset
                or stop <= start or stop > len(probabilities)):
            raise ValueError(f"记录窗口区间缺失、重叠或越界：{key}")
        seen.add(key)
        result.append({**record, "probabilities": probabilities[start:stop].mean(axis=0).tolist(),
                       "window_count": stop - start})
        offset = stop
    if offset != len(probabilities):
        raise ValueError("记录区间未覆盖全部窗口概率")
    return result


def aggregate_units(record_predictions):
    """PU 按 20 条记录（KA08 特例 19 条），CW 按 DE/FE 两记录等权聚合。"""
    groups, seen = {}, set()
    for record in record_predictions:
        key = record["record_key"]
        if key in seen:
            raise ValueError(f"记录重复：{key}")
        seen.add(key)
        label = record["label"]
        if label is not None and (type(label) is not int or label not in range(4)):
            raise ValueError(f"记录标签无效：{key}")
        if type(record["window_count"]) is not int or record["window_count"] <= 0:
            raise ValueError(f"记录窗口数无效：{key}")
        groups.setdefault(record["unit_key"], []).append(record)
    result = []
    for unit, rows in groups.items():
        first = rows[0]
        if any(any(row.get(field) != first.get(field) for field in UNIT_FIELDS) for row in rows):
            raise ValueError(f"单元内标签或元数据不一致：{unit}")
        channels = Counter(row["channel"] for row in rows)
        if first["dataset"] == "paderborn":
            expected = 19 if unit == "KA08/N15_M01_F10" else 20
            if (unit != f"{first['bearing_id']}/{first['condition']}"
                    or channels != Counter({"vibration_1": expected})
                    or len({str(row['path']) for row in rows}) != expected):
                raise ValueError(f"PU 单元记录缺失、重复或通道错误：{unit}")
        elif first["dataset"] == "cwru":
            if (channels != Counter({"DE": 1, "FE": 1}) or first["label"] is None
                    or any(unit != row['source'] + '/' + row['path'].replace('\\', '/').rsplit('/', 1)[-1]
                           or row["record_key"] != f"{unit}::{row['channel']}" for row in rows)):
                raise ValueError(f"CWRU 单元记录缺失或通道错误：{unit}")
        else:
            raise ValueError(f"未知数据集：{first['dataset']}")
        probabilities = _probabilities([row["probabilities"] for row in rows])
        result.append({**{field: first.get(field) for field in UNIT_FIELDS},
                       "channels": sorted(channels), "probabilities": probabilities.mean(axis=0).tolist(),
                       "record_count": len(rows), "window_count": sum(row["window_count"] for row in rows)})
    return result


def _classification_metrics(labels, predicted):
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, guess in zip(labels, predicted):
        confusion[truth, guess] += 1
    support = confusion.sum(axis=1)
    recall, f1 = [], []
    for index in range(4):
        tp = int(confusion[index, index])
        recall.append(tp / int(support[index]) if support[index] else None)
        denominator = int(support[index] + confusion[:, index].sum())
        f1.append(2 * tp / denominator if support[index] else None)
    present = [index for index in range(4) if support[index]]
    macro_recall = float(np.mean([recall[index] for index in present])) if present else None
    return {"n": len(labels), "accuracy": float(np.trace(confusion) / len(labels)) if len(labels) else None,
            "macro_f1": float(np.mean([f1[index] for index in present])) if present else None,
            "macro_recall": macro_recall, "recall": recall,
            "confusion": confusion.tolist(), "support": support.tolist(),
            "classes": CLASS_NAMES, "averaged_classes": [CLASS_NAMES[index] for index in present]}


def _score(rows):
    labeled = [row for row in rows if row["label"] is not None]
    if any(type(row["label"]) is not int or row["label"] not in range(4) for row in labeled):
        raise ValueError("评分标签必须是 0/1/2/3 或 None")
    probabilities = (_probabilities([row["probabilities"] for row in labeled])
                     if labeled else np.empty((0, 4)))
    labels = [row["label"] for row in labeled]
    e1 = probabilities.argmax(axis=1) if labeled else np.array([], dtype=int)
    masked = probabilities.copy()
    masked[:, 1] = 0
    e2 = masked.argmax(axis=1) if labeled else np.array([], dtype=int)
    return {"n": len(labeled), "total_count": len(rows), "unlabeled_count": len(rows) - len(labeled),
            "E1": _classification_metrics(labels, e1), "E2": _classification_metrics(labels, e2),
            "P_ball": float(np.mean(e1 == 1)) if labeled else None,
            "denominator": "有真实标签的全部样本；E1/E2/P_ball 使用相同分母"}


def score_predictions(predictions, pool_units=None):
    """真实支持类别宏平均；PU 按工况、来源及监督池轴承是否见过并报。"""
    result = _score(predictions)
    primary = [row for row in predictions if row.get("dataset") == "paderborn" and row["label"] is not None]
    source = [row for row in predictions if row.get("dataset") == "cwru"]
    result["source"] = _score(source)
    result["source_by_source"] = {
        name: _score([row for row in source if row["source"] == name])
        for name in sorted({row["source"] for row in source})}
    result["source_by_channel"] = {
        name: _score([row for row in source if row.get("channel") == name])
        for name in sorted({row["channel"] for row in source if "channel" in row})}
    result["groups"] = {}
    for field in ("condition", "origin", "severity"):
        values = sorted({row.get(field) for row in primary}, key=str)
        result["groups"][field] = {str(value): _score([row for row in primary if row.get(field) == value])
                                   for value in values}
    if pool_units is None:
        result["groups"]["seen_unseen_bearing"] = {"unknown": _score(primary)}
    else:
        seen = {unit.split("/", 1)[0] for unit in pool_units}
        result["groups"]["seen_unseen_bearing"] = {
            name: _score([row for row in primary if (row["bearing_id"] in seen) == is_seen])
            for name, is_seen in (("seen", True), ("unseen", False))}
    kb = [row for row in predictions if row["label"] is None]
    kb_report = {"n": len(kb), "prediction_counts": [0] * 4, "prediction_fractions": None,
                 "by_bearing": {}, "note": "KB 无单类真值，不入 E1/E2/P_ball 分母；官方主导指派仅作参考"}
    references = {"KB23": 2, "KB24": 2, "KB27": 3}
    for bearing in sorted({row.get("bearing_id") for row in kb}, key=str):
        rows = [row for row in kb if row.get("bearing_id") == bearing]
        predicted = _probabilities([row["probabilities"] for row in rows]).argmax(axis=1)
        counts = np.bincount(predicted, minlength=4)
        reference = references.get(bearing)
        kb_report["by_bearing"][str(bearing)] = {
            "n": len(rows), "prediction_counts": counts.tolist(),
            "prediction_fractions": (counts / len(rows)).tolist(),
            "official_dominant_reference": reference,
            "reference_agreement": float(np.mean(predicted == reference)) if reference is not None else None,
            "normalized_entropy": _entropy([row["probabilities"] for row in rows]).tolist(),
            "normalized_entropy_mean": float(_entropy([row["probabilities"] for row in rows]).mean())}
        kb_report["prediction_counts"] = (np.asarray(kb_report["prediction_counts"]) + counts).tolist()
    if kb:
        kb_report["prediction_fractions"] = (np.asarray(kb_report["prediction_counts"]) / len(kb)).tolist()
    result["KB"] = kb_report
    return result


def infer_records(model, feature, batch_size=256, device=None, inputs=None):
    """只读前向：显式 eval/no_grad，检查所有 BN buffers 未改变。

    inputs 为 None 时用 feature.X；双通道模型传入 features.dual_input 组装的 (N,2,256)。
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device
    if inputs is None:
        inputs = feature.X
    inputs = torch.as_tensor(inputs)
    if inputs.ndim != 3 or len(inputs) != len(feature.X):
        raise ValueError(f"前向输入必须是 (N,C,256) 且与窗口数一致，实际 {tuple(inputs.shape)}")
    model.to(device)
    model.eval()
    bn_before = {f"{name}.{key}": value.detach().clone()
                 for name, module in model.named_modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                 for key, value in module.named_buffers()}
    output = []
    try:
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start:start + batch_size].to(device)
                logits = model(batch)
                # 复用网络默认返回 (logits, attention)，小模型也可直接返回 logits。
                if isinstance(logits, tuple):
                    logits = logits[0]
                if (not isinstance(logits, torch.Tensor) or logits.shape != (len(batch), 4)
                        or not torch.isfinite(logits).all()):
                    raise ValueError("模型必须输出有限的 (batch,4) logits")
                output.append(torch.softmax(logits, dim=1).cpu().numpy())
    finally:
        bn_after = {f"{name}.{key}": value for name, module in model.named_modules()
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    for key, value in module.named_buffers()}
        if bn_before.keys() != bn_after.keys() or any(
                not torch.equal(value, bn_after[key]) for key, value in bn_before.items()):
            raise RuntimeError("测试前向改变了 BN buffers，拒绝报告")
    probabilities = np.concatenate(output) if output else np.empty((0, 4))
    return aggregate_records(feature.records, probabilities)


def _write_predictions(rows, path, unit=False):
    fields = (["unit_key"] if unit else ["record_key", "unit_key", "path", "channel", "start", "stop"])
    fields += ["dataset", "source", "bearing_id", "condition", "origin", "severity", "src_fs", "label",
               "prob_Normal", "prob_Ball", "prob_IR", "prob_OR", "prediction_E1", "prediction_E2", "window_count"]
    fields += [name for name in ("logspec_window_count", "e_ir", "e_or", "anomaly_mean", "normal_threshold_fraction", "anomaly_quantiles")
               if any(name in row for row in rows)]
    if unit:
        fields += ["record_count", "channels"]
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            probabilities = _probabilities([row["probabilities"]])[0]
            masked = probabilities.copy()
            masked[1] = 0
            values = {**row, **dict(zip(("prob_Normal", "prob_Ball", "prob_IR", "prob_OR"),
                                        probabilities.tolist())),
                      "prediction_E1": int(probabilities.argmax()), "prediction_E2": int(masked.argmax())}
            if "anomaly_quantiles" in values:
                values["anomaly_quantiles"] = json.dumps(values["anomaly_quantiles"])
            if unit:
                values["channels"] = ";".join(row.get("channels", []))
            writer.writerow(values)


def read_predictions(path, unit=False):
    """把 predictions.csv/unit_predictions.csv 读回带 probabilities 的普通行字典。"""
    path = Path(path)
    nullable = {"label", "bearing_id", "condition", "origin", "severity"}
    integers = {"window_count", "record_count", "src_fs", "prediction_E1", "prediction_E2"}
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in ("prob_Normal", "prob_Ball", "prob_IR", "prob_OR"):
                    row["probabilities"] = row.get("probabilities", []) + [float(value)]
                elif key in nullable:
                    # 空单元 → None；纯数字列（label/severity）转 int，其余保留字符串。
                    row[key] = None if value == "" else (
                        int(value) if value.lstrip("-").isdigit() else value)
                elif key in ("start", "stop"):
                    # 旧完整记录锚的 CSV 这两列为空字符串；其余整数列仍必须可解析。
                    row[key] = None if value == "" else int(value)
                elif key in integers or key == "logspec_window_count":
                    row[key] = int(value)
                elif key in ("e_ir", "e_or", "anomaly_mean", "normal_threshold_fraction"):
                    row[key] = float(value)
                elif key == "anomaly_quantiles":
                    row[key] = json.loads(value)
                elif key == "channels":
                    row[key] = value.split(";") if value else []
                else:
                    row[key] = value
            if len(row.get("probabilities", [])) != 4:
                raise ValueError(f"CSV 行概率列不完整：{path}")
            rows.append(row)
    if rows:
        _probabilities([row["probabilities"] for row in rows])
    return rows


def save_reports(record_predictions, unit_predictions, output, pool_units=None, eval_info=None):
    """仅创建四份报告；允许目录已有 rule.json 等额外文件，不覆盖已有同名产物。"""
    output = Path(output)
    names = ("metrics.json", "predictions.csv", "unit_predictions.csv", "eval_info.json")
    for name in names:
        if (output / name).exists():
            raise FileExistsError(f"报告已存在，拒绝覆盖：{output / name}")
    info = dict(eval_info or {})
    info.setdefault("protocol", "v8_stage3")
    info["inference"] = "eval/no_grad；无训练、梯度更新或 BN 重估"
    info["supervised_pool_units"] = None if pool_units is None else sorted(pool_units)
    metrics = {"record_level": score_predictions(record_predictions, pool_units),
               "unit_level": score_predictions(unit_predictions, pool_units)}
    primary = [row for row in unit_predictions if row.get("dataset") == "paderborn" and row["label"] is not None]
    primary_metrics = _score(primary)
    e1, e2 = primary_metrics["E1"]["macro_recall"], primary_metrics["E2"]["macro_recall"]
    full = len(primary) == 116 and len({row["unit_key"] for row in primary}) == 116
    metrics["D3"] = {"n": len(primary), "required_n": 116, "E1": e1, "E2": e2,
                     "E1_min": 0.8, "E2_min": 0.9,
                     "passed": bool(e1 >= 0.8 and e2 >= 0.9) if full else None,
                     "literature_anchor_E2": 0.75,
                     "interpretation": info.get("evaluation_scope", "监督池未知，不能判为泛化证据"),
                     "note": "全 116 单元真实支持类宏平均；含监督池时仅描述，K116 仅 in_sample"}
    for name, units in info.get("evaluation_splits", {}).items():
        wanted = set(units)
        overlap = len(wanted & set(pool_units or []))
        metrics.setdefault("splits", {})[name] = {
            "requested_unit_count": len(wanted), "supervised_pool_overlap_unit_count": overlap,
            "interpretation": "no_heldout" if not wanted else (
                "source_validation" if name == "source_validation" else (
                    info.get("evaluation_scope", "监督池未知") if name in ("all_primary", "source_all") else (
                        "descriptive_contains_training_units" if overlap else (
                            "transductive_labels_heldout_signals_seen" if info.get("transductive") else
                            info.get("heldout_scope", "natural_heldout"))))),
            "unit_level": score_predictions([row for row in unit_predictions if row["unit_key"] in wanted], pool_units),
            "record_level": score_predictions([row for row in record_predictions if row["unit_key"] in wanted], pool_units)}
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("metrics.json", metrics), ("eval_info.json", info)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    _write_predictions(record_predictions, output / "predictions.csv")
    _write_predictions(unit_predictions, output / "unit_predictions.csv", unit=True)
    return metrics


# ---------------------------------------------------------------------------
# 阶段三公共部分：配置/清单校验与评估口径信息
# ---------------------------------------------------------------------------

def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"冻结产物缺失：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _project(value):
    """配置里的相对路径相对 v8 根解析。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _output_dir(value):
    """正式评估产物只允许写入 v8/outputs；测试用 mock ROOT 指向临时目录。"""
    output = _project(value).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError(f"评估产物只允许写入 v8/outputs：{output}")
    return output


def _check_pilot(metadata, kind, where):
    """正式评估拒绝 pilot 产物；P4 只做源诊断，不冒充正式格。"""
    if metadata.get("run_kind") != "formal":
        raise ValueError(f"{kind} 不是正式运行（run_kind={metadata.get('run_kind')!r}）：{where}")


def _validate_fusion_config(config):
    fusion = config.get("fusion")
    if not isinstance(fusion, dict):
        raise ValueError("配置缺少冻结 fusion 块")
    weights = fusion.get("weights")
    if (not isinstance(weights, list) or {tuple(map(float, w)) for w in weights} != FROZEN_WEIGHTS
            or any(len(w) != 3 or abs(sum(w) - 1) > 1e-9 or w[0] < 0.5 for w in weights)):
        raise ValueError("fusion.weights 偏离冻结六候选（w_D>=0.5，和为1）")
    if fusion.get("weight_names") != ["deep", "normal", "physical"]:
        raise ValueError("fusion.weight_names 必须是 deep/normal/physical")
    if fusion.get("tie_break") != FROZEN_TIE_BREAK:
        raise ValueError("fusion.tie_break 偏离冻结平票规则")
    if fusion.get("uncertainty_quantile") != 0.05 or fusion.get("quantile_method") != "linear":
        raise ValueError("fusion 不确定分位必须为 0.05/linear")
    if fusion.get("ablations") != FROZEN_ABLATIONS:
        raise ValueError("fusion.ablations 必须是 without_phys/without_normal")


def _paired_splits(config):
    """读取并校验配对清单；直接复用 selection.paired_splits 的冻结补集关系。

    自然留出只避开各自池：T_old=U-P_old、T_gg=U-P_gg，只有 T_pair 避开两池并集。
    """
    from selection import paired_splits
    paired = _read_json(_project(config["paired_splits_path"]))
    expected = paired_splits(_read_json(_project(config["selection_path"])),
                             _read_json(_project(config["selection_gg_path"])))
    for name, value in expected.items():
        actual = paired.get(name)
        if (not isinstance(actual, list) or len(actual) != len(value)
                or set(actual) != set(value)):
            raise ValueError(f"配对清单 {name} 与冻结新旧池不一致：{config['paired_splits_path']}")
    return {"all_primary": list(expected["U"]), "T_old": list(expected["T_old"]),
            "T_gg": list(expected["T_gg"]), "T_pair": list(expected["T_pair"]),
            "common_unseen_units": list(expected["common_unseen_units"])}


def _stage3_scope(arm):
    """臂到（池种类, 评估口径, 是否传导）的固定映射。"""
    if arm in ("joint", "joint_logspec", "mixup4src", "coral4src", "phys_envelope", "phys_ies",
               "fusion_joint", "fusion_joint_without_phys", "fusion_joint_without_normal"):
        return [], "zero_shot_all_primary", False
    if arm == "adabn":
        return [], "transductive_test_signals_seen_labels_unused", True
    if arm in ("ce_K19", "ce_K19_logspec", "svm_phys", "fusion_ce_K19",
               "fusion_ce_K19_without_phys", "fusion_ce_K19_without_normal"):
        return "K19_old", "descriptive_contains_training_units", False
    if arm == "ce_K19_gg":
        return "K19_gg", "descriptive_contains_training_units", False
    if arm == "ce_K116":
        return "K116", "in_sample_only_no_heldout", False
    raise ValueError(f"未知评估臂：{arm}")


def _pool_units_for(config, arm):
    selection = _read_json(_project(config["selection_path"]))
    if arm == "ce_K19_gg":
        return _read_json(_project(config["selection_gg_path"]))["budgets"]["K19"]["pool"]
    if arm == "ce_K116":
        return selection["budgets"]["K116"]["pool"]
    if arm in ("ce_K19", "ce_K19_logspec", "svm_phys", "fusion_ce_K19",
               "fusion_ce_K19_without_phys", "fusion_ce_K19_without_normal"):
        return selection["budgets"]["K19"]["pool"]
    return []


def _source_split(cache, fold, config):
    from datasets.cwru import split_source_units
    if fold not in config["folds"]:
        raise ValueError("fold 不在冻结矩阵中")
    return split_source_units([SimpleNamespace(**row) for row in cache.records], config["split_seed"])[fold]


def _stage3_info(config, arm, dataset, cache, pool_units, fold=None, seed=None, extra=None):
    """构造 eval_info：协议、口径、冻结 splits（PU 用配对清单，CW 用源折）。"""
    pool_kind, scope, transductive = _stage3_scope(arm)
    if cache.version != config["cache_version"]:
        raise ValueError(f"缓存口径 {cache.version} 与冻结配置不一致")
    pools = {"K19_old": _read_json(_project(config["selection_path"]))["budgets"]["K19"]["pool"],
             "K19_gg": _read_json(_project(config["selection_gg_path"]))["budgets"]["K19"]["pool"],
             "K116": _read_json(_project(config["selection_path"]))["budgets"]["K116"]["pool"]}
    pool = list(pools[pool_kind]) if isinstance(pool_kind, str) else list(pool_kind)
    if pool_units is not None and set(pool_units) != set(pool):
        raise ValueError(f"监督池与冻结口径不一致：{arm}")
    if dataset == "cwru":
        split = _source_split(cache, fold, config)
        splits = {"source_all": sorted({row["unit_key"] for row in cache.records}),
                  "source_validation": sorted(split["val"])}
        scope = "source_all_descriptive_contains_training_units"
        heldout_scope = "source_validation"
    else:
        splits = _paired_splits(config)
        if pool_kind == "K116":
            heldout_scope = "no_heldout"  # 只有 K116 没有留出
        elif transductive:
            heldout_scope = "transductive_labels_heldout_signals_seen"
        elif pool:
            heldout_scope = "supervised_pool_outside"
        else:
            heldout_scope = "zero_shot"  # 零监督池是零样本，不是 no_heldout
    info = {"protocol": "v8_stage3", "arm": arm, "fold": fold, "seed": seed,
            "dataset": dataset, "cache_version": cache.version, "repr_config": cache.repr_config,
            "run_kind": "formal", "evaluation_scope": scope, "transductive": transductive,
            "unlabeled_target_label_usage": 0, "supervised_pool_unit_count": len(pool),
            "heldout_scope": heldout_scope, "evaluation_splits": splits}
    info.update(extra or {})
    return info, pool


# ---------------------------------------------------------------------------
# 深度评估：checkpoint 校验与双通道只读前向
# ---------------------------------------------------------------------------

def _check_checkpoint(checkpoint, config, arm, fold, seed, cache):
    from preprocessing import config_from_saved
    if not isinstance(checkpoint, dict) or checkpoint.get("protocol") != "v8_stage3":
        raise ValueError("只接受 v8_stage3 checkpoint")
    if checkpoint.get("arm") != arm or checkpoint.get("fold") != fold or checkpoint.get("seed") != seed:
        raise ValueError(f"checkpoint 臂/折/种子与评估命令不一致："
                         f"{checkpoint.get('arm')}/{checkpoint.get('fold')}/{checkpoint.get('seed')}")
    if checkpoint.get("checkpoint_kind") != "last":
        raise ValueError("正式评估只接受 last checkpoint，不接受 best")
    epochs = config["parameters"]["epochs"]
    for field in ("epoch", "checkpoint_epoch", "last_epoch"):
        if checkpoint.get(field) != epochs:
            raise ValueError(f"正式评估只接受完成 {epochs} 轮的 last；{field} 必须为 {epochs}")
    _check_pilot(checkpoint, "checkpoint", "last.pt")
    if checkpoint.get("cache_version") != config["cache_version"]:
        raise ValueError("checkpoint 缓存口径与冻结配置不一致")
    if config_from_saved(checkpoint.get("repr_config"), require_complete=True) != cache.repr_config:
        raise ValueError("checkpoint 表征与冻结缓存不一致")
    model_config = checkpoint.get("model")
    if (not isinstance(model_config, dict) or model_config.get("n_channels") != 2
            or model_config.get("n_classes") != 4 or model_config.get("frontend", "spectral_v1") != "spectral_v1"):
        raise ValueError("只允许固定的双通道四类 spectral_v1 网络")
    if not isinstance(checkpoint.get("source_split"), dict):
        raise ValueError("checkpoint 必须携带训练时的源折划分 source_split")
    scaler = checkpoint.get("scaler")
    if (not isinstance(scaler, dict) or not isinstance(scaler.get("mean"), list)
            or not isinstance(scaler.get("std"), list)
            or len(scaler["mean"]) != 25 or len(scaler["std"]) != 25):
        raise ValueError("checkpoint 必须携带 25 维源训练 scaler（mean/std 列表）")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("checkpoint scaler 的 mean/std 必须有限且 std>0")
    return model_config, scaler


def _check_supervised_pool(checkpoint, config, arm):
    """监督池字段必须显式给出（空池也须 []）；缺失或 None 不得绕过冻结口径。"""
    pool_field = checkpoint.get("supervised_pool_units")
    frozen = _pool_units_for(config, arm)
    if type(pool_field) is not list or set(pool_field) != set(frozen):
        raise ValueError(f"checkpoint 监督池字段缺失、None 或与冻结口径不一致：{arm}")
    return pool_field


def _run_deep(config, args, execute):
    if not args.checkpoint:
        raise ValueError("深度评估必须指定 --checkpoint .../last.pt")
    if not execute:
        # dry-run：臂/折/种子可能由 checkpoint 提供，此处只报命令与依赖存在性。
        print(json.dumps({"arm": args.arm, "fold": args.fold, "seed": args.seed,
                          "dataset": args.dataset, "execute": False,
                          "output": str(_output_dir(args.output)) if args.output else None,
                          "checkpoint_exists": Path(args.checkpoint).is_file()},
                         ensure_ascii=False))
        return
    import features
    from features import feature_subset, load_cache
    from model import MSRSABiLSTM
    if args.dataset is None:
        raise ValueError("深度评估必须指定 --dataset cwru/paderborn")
    cache = load_cache(args.dataset, _project(config["cache_dir"]))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # 臂/折/种子默认取自 checkpoint；命令行显式给出时必须一致。
    arm = args.arm if args.arm is not None else checkpoint.get("arm")
    fold = args.fold if args.fold is not None else checkpoint.get("fold")
    seed = args.seed if args.seed is not None else checkpoint.get("seed")
    if arm not in config["deep_arms"]:
        raise ValueError(f"{arm} 不是深度评估臂")
    if fold not in config["folds"] or seed not in config["seeds"]:
        raise ValueError(f"checkpoint 折/种子不在冻结矩阵中：{fold}/{seed}")
    if args.arm is not None and args.arm != arm:
        raise ValueError(f"--arm {args.arm} 与 checkpoint 臂 {arm} 不一致")
    model_config, scaler = _check_checkpoint(checkpoint, config, arm, fold, seed, cache)
    pool_field = _check_supervised_pool(checkpoint, config, arm)
    model = MSRSABiLSTM(**{name: model_config[name] for name in
                           ("n_channels", "n_classes", "lstm_hidden", "use_se") if name in model_config})
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    feature = feature_subset(cache, labeled=False)
    # logspec-only 对照：同一个双通道模型，第二通道缩放后恒零；补零在缩放之后。
    logspec_only = arm.endswith("_logspec")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = features.dual_input(feature.X, feature.phys25, scaler, logspec_only=logspec_only)
    records = infer_records(model, feature, device=device, inputs=inputs)
    units = aggregate_units(records)
    output = _output_dir(args.output or ROOT / "outputs/eval" / f"{arm}_f{fold}_s{seed}_{args.dataset}")
    provenance = {"checkpoint": str(Path(args.checkpoint).resolve()),
                  "checkpoint_kind": checkpoint.get("checkpoint_kind"),
                  "epoch": checkpoint.get("epoch"), "last_epoch": checkpoint.get("last_epoch"),
                  "repr_config": checkpoint.get("repr_config"), "scaler": scaler,
                  "source_split": checkpoint.get("source_split"),
                  "supervised_pool_units": pool_field}
    extra = {"checkpoint": str(Path(args.checkpoint).resolve()),
             "logspec_only": logspec_only,
             "scaler_source": "源外层折训练窗口拟合，随 checkpoint 落盘",
             "provenance": provenance}
    # 该深度预测器自己的源验证单元 tau；主 PU/KB 分开报覆盖与熵，不改主分母。
    if args.dataset == "paderborn":
        source_cache = load_cache("cwru", _project(config["cache_dir"]))
        val_feature = feature_subset(source_cache,
                                     sorted(_source_split(source_cache, fold, config)["val"]),
                                     labeled=False)
        val_records = infer_records(model, val_feature, device=device,
                                    inputs=features.dual_input(val_feature.X, val_feature.phys25, scaler,
                                                               logspec_only=logspec_only))
        extra["uncertainty"] = _unit_uncertainty(
            [row["probabilities"] for row in aggregate_units(val_records)], units,
            [row["probabilities"] for row in units], config["fusion"]["uncertainty_quantile"])
    info, pool = _stage3_info(config, arm, args.dataset, cache, pool_units=pool_field,
                              fold=fold, seed=seed, extra=extra)
    metrics = save_reports(records, units, output, pool_units=pool, eval_info=info)
    print(f"评估已落盘：{output}，记录分母 {metrics['record_level']['n']}，单元分母 {metrics['unit_level']['n']}")


# ---------------------------------------------------------------------------
# 浅层评估：复用 shallow.predict_records，评估前核模型配置与拟合来源
# ---------------------------------------------------------------------------

def _check_auxiliary_model(config, model_dir, arm, fold, seed, source_cache, selection):
    """读 joblib 前核 config/split：arm、fold、seed、版本、run_kind 与拟合来源一致。"""
    model_dir = Path(model_dir)
    metadata = _read_json(model_dir / "config.json")
    if metadata.get("protocol") != "v8_stage3" or metadata.get("arm") != arm:
        raise ValueError(f"模型产物不是阶段三 {arm}：{model_dir}")
    if metadata.get("fold") != fold or metadata.get("seed") != seed:
        raise ValueError(f"模型产物 fold/seed 与评估命令不一致："
                         f"{metadata.get('fold')}/{metadata.get('seed')}")
    if metadata.get("cache_version") != config["cache_version"]:
        raise ValueError("模型产物缓存口径与冻结配置不一致")
    _check_pilot(metadata, "模型", model_dir)
    split = _read_json(model_dir / "split.json")
    actual = _source_split(source_cache, fold, config)
    source_part = split if arm == "normal" else split.get("source", {})
    if (set(source_part.get("train", [])) != set(actual["train"])
            or set(source_part.get("val", [])) != set(actual["val"])):
        raise ValueError(f"模型源折划分与当前冻结折不一致：{model_dir}")
    if arm == "normal":
        if seed != 0:
            raise ValueError("Normal 跨深度 seed 复用，评估必须使用 seed=0")
    elif set(split.get("target", {}).get("pool", [])) != set(selection["budgets"]["K19"]["pool"]):
        raise ValueError(f"模型目标池与冻结旧 K19 不一致：{model_dir}")
    return metadata


def _run_shallow(config, args, execute):
    arm, fold, seed = args.arm, args.fold, args.seed
    if arm != "svm_phys":
        raise ValueError(f"{arm} 不是浅层评估臂")
    if fold is None or seed is None:
        raise ValueError("浅层评估必须显式指定 --fold/--seed")
    if args.dataset is None:
        raise ValueError("浅层评估必须指定 --dataset cwru/paderborn")
    model_dir = args.model_dir or f"outputs/train/svm_phys_f{fold}_s{seed}"
    output = _output_dir(args.output or ROOT / "outputs/eval" / f"{arm}_f{fold}_s{seed}_{args.dataset}")
    if not execute:
        needed = ["model.joblib", "scaler.json", "config.json", "split.json", "calibration.json",
                  "fit_info.json", "quotas.json"]
        print(json.dumps({"arm": arm, "fold": fold, "seed": seed, "dataset": args.dataset,
                          "execute": False, "output": str(output), "model_dir": str(model_dir),
                          "missing_files": [n for n in needed if not (Path(model_dir) / n).is_file()]},
                         ensure_ascii=False))
        return
    from features import feature_subset, load_cache
    import shallow
    cache = load_cache(args.dataset, _project(config["cache_dir"]))
    source_cache = cache if args.dataset == "cwru" else load_cache("cwru", _project(config["cache_dir"]))
    selection = _read_json(_project(config["selection_path"]))
    _check_auxiliary_model(config, model_dir, arm, fold, seed, source_cache, selection)
    records = shallow.predict_records(cache, model_dir)
    if not records:
        raise ValueError("浅层预测为空，拒绝写报告")
    units = aggregate_units(records)
    extra = {"model_dir": str(Path(model_dir).resolve()),
             "calibration": "独立源验证单元留出 sigmoid 校准；校准后分数不再作为独立源泛化得分"}
    # 该浅层预测器自己的源验证单元 tau；主 PU/KB 分开报覆盖与熵，不改主分母。
    if args.dataset == "paderborn":
        val_cache = feature_subset(source_cache,
                                   sorted(_source_split(source_cache, fold, config)["val"]),
                                   labeled=False)
        val_units = aggregate_units(shallow.predict_records(val_cache, model_dir))
        extra["uncertainty"] = _unit_uncertainty(
            [row["probabilities"] for row in val_units], units,
            [row["probabilities"] for row in units], config["fusion"]["uncertainty_quantile"])
    info, pool = _stage3_info(config, arm, args.dataset, cache,
                              pool_units=_pool_units_for(config, arm),
                              fold=fold, seed=seed, extra=extra)
    metrics = save_reports(records, units, output, pool_units=pool, eval_info=info)
    print(f"评估已落盘：{output}，记录分母 {metrics['record_level']['n']}，单元分母 {metrics['unit_level']['n']}")


# ---------------------------------------------------------------------------
# 晚期融合：权重选择、两路消融、tau 与熵报告
# ---------------------------------------------------------------------------

def _q_phys_record_rows(cache, records):
    """逐记录 q_phys 软证据；窗口数为物理分析窗数，与 logspec 窗数不混用。

    完整缓存用记录自身 analysis 区间；feature_subset 共享全量 ies_analysis，
    必须按 analysis_ranges 的原始区间取窗。
    """
    from physical_baseline import ies_evidence
    analysis = np.asarray(cache.ies_analysis, dtype=np.float64)
    if analysis.ndim != 2 or analysis.shape[1] != 16:
        raise ValueError("缓存 ies_analysis 必须是 (A,16) 原始 R")
    ranges = getattr(cache, "analysis_ranges", None)
    if ranges is not None and len(ranges) != len(records):
        raise ValueError("analysis_ranges 与记录数不一致")
    rows = []
    for index, record in enumerate(records):
        if ranges is not None:
            start, stop = int(ranges[index][0]), int(ranges[index][1])
        else:
            start, stop = record["analysis_start"], record["analysis_stop"]
        if (type(start) is not int or type(stop) is not int or start < 0 or stop <= start
                or stop > len(analysis)):
            raise ValueError(f"记录物理分析区间缺失或越界：{record['record_key']}")
        _, q_phys, _, _ = ies_evidence(analysis[start:stop])
        q_phys = np.asarray(q_phys, dtype=np.float64)
        if (q_phys.shape != (stop - start, 4) or not np.isfinite(q_phys).all()
                or (q_phys <= 0).any() or not np.allclose(q_phys.sum(axis=1), 1, atol=1e-6)):
            raise ValueError(f"q_phys 必须为逐窗正值归一四类概率：{record['record_key']}")
        rows.append({**record, "probabilities": q_phys.mean(axis=0).tolist(),
                     "window_count": stop - start})
    return rows


def _macro_recall(labels, probabilities):
    return _classification_metrics(labels, np.asarray(probabilities).argmax(axis=1))["macro_recall"]


def _align_units(units_list, names):
    """多路单元证据按 unit_key 对齐；缺一路记录直接报错，不做静默交集。"""
    aligned, reference = [], None
    for name, rows in zip(names, units_list):
        index = {}
        for row in rows:
            if row["unit_key"] in index:
                raise ValueError(f"{name} 单元重复：{row['unit_key']}")
            index[row["unit_key"]] = row
        if reference is None:
            reference = index
        elif set(index) != set(reference):
            missing = sorted(set(reference) - set(index))[:3]
            extra = sorted(set(index) - set(reference))[:3]
            raise ValueError(f"{name} 单元清单与深度不一致：缺 {missing}，多 {extra}")
        aligned.append(index)
    keys = sorted(reference)
    for name, index in zip(names[1:], aligned[1:]):
        for key in keys:
            if index[key]["label"] != reference[key]["label"]:
                raise ValueError(f"{name} 与深度单元真值不一致：{key}")
    return keys, [reference[key]["label"] for key in keys], aligned


def _select_weights(deep_units, normal_units, phys_units, candidates):
    """只在源验证单元四类宏召回（E1）上最大化；精确同分依次比 w_D、辅助支路数、w_N。"""
    keys, labels, (d, n, p) = _align_units([deep_units, normal_units, phys_units],
                                           ("deep", "normal", "q_phys"))
    matrices = [np.asarray([index[key]["probabilities"] for key in keys], dtype=np.float64)
                for index in (d, n, p)]
    report, best = [], None
    for weights in candidates:
        fused = weights[0] * matrices[0] + weights[1] * matrices[1] + weights[2] * matrices[2]
        score = _macro_recall(labels, fused)
        report.append({"weights": [float(w) for w in weights], "source_val_macro_recall_E1": score})
        # 平票优先级：w_D 较大 → 辅助非零支路较少 → w_N 较大。
        key = (score, weights[0], -sum(w > 0 for w in weights[1:]), weights[1])
        if best is None or key > best[0]:
            best = (key, [float(w) for w in weights], score)
    return {"chosen_weights": best[1], "chosen_source_val_macro_recall_E1": best[2],
            "candidates": report, "unit_count": len(keys)}, (keys, labels, matrices)


def _renormalize(weights, drop):
    """两路消融：置零相应支路后重新归一化；w_D>=0.5 保证分母非零。"""
    reduced = [float(w) for w in weights]
    reduced[2 if drop == "without_phys" else 1] = 0.0
    total = sum(reduced)
    if total <= 0:
        raise ValueError("两路消融分母非零，不应发生")
    return [w / total for w in reduced]


def _entropy(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    terms = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -(terms.sum(axis=1) / np.log(4))


def _unit_uncertainty(val_prob, target_units, target_prob, quantile=0.05):
    """单版本不确定报告：tau=Q_quantile(源验证单元 max p)，主 PU/KB 分报覆盖与熵。

    max(p_unit)<tau 只标记低置信，不拒判、不改类、不删样本、不改 E1/E2 分母。
    """
    val_max = _probabilities(val_prob).max(axis=1)
    if not len(val_max):
        raise ValueError("源验证单元概率为空，无法确定 tau")
    target_prob = _probabilities(target_prob)
    if len(target_prob) != len(target_units):
        raise ValueError("目标单元概率行数与单元清单不一致")
    tau = float(np.quantile(val_max, quantile, method="linear"))
    scopes = {}
    for scope, labeled in (("main_pu", True), ("kb", False)):
        picked = [index for index, row in enumerate(target_units)
                  if (row["label"] is not None) == labeled]
        if not picked:
            scopes[scope] = {"unit_count": 0, "coverage": None,
                             "low_confidence_unit_count": 0, "low_confidence_units": [],
                             "normalized_entropy_by_unit": {}}
            continue
        probabilities = target_prob[picked]
        maximum, entropy = probabilities.max(axis=1), _entropy(probabilities)
        scopes[scope] = {
            "unit_count": len(picked), "coverage": float(np.mean(maximum >= tau)),
            "low_confidence_unit_count": int(np.sum(maximum < tau)),
            "low_confidence_units": [target_units[index]["unit_key"] for index, value
                                     in zip(picked, maximum) if value < tau],
            "normalized_entropy_by_unit": {target_units[index]["unit_key"]: float(value)
                                           for index, value in zip(picked, entropy)}}
    return {"tau": tau, "quantile": quantile, "quantile_method": "linear",
            "source_val_unit_count": len(val_max),
            "source_val_coverage": float(np.mean(val_max >= tau)), "scopes": scopes,
            "note": "max(p_unit)<tau 仅标记低置信；不拒判、不改类、不删分母；主 PU 与 KB 分开报告"}


def _uncertainty_report(versions_prob, target_units, quantile=0.05):
    """每个版本用自己的源验证概率定 tau；主 PU/KB 分别报覆盖、低置信名单与逐单元熵。"""
    return {name: _unit_uncertainty(val_prob, target_units, target_prob, quantile)
            for name, (val_prob, target_prob) in versions_prob.items()}


def _fuse_records(weights, deep_rows, normal_rows=None, phys_rows=None):
    """记录级用相同固定权重线性融合；返回与 deep 行字段一致的融合行。

    先按权重求和，未用的支路权重为零即自然跳过；两条计算路径（先记录后单元
    与直接单元级加权）的线性一致性由测试保证。
    """
    aux = [(w, rows) for w, rows in zip(weights[1:], (normal_rows, phys_rows)) if w > 0]
    for w, rows in aux:
        if rows is None or len(rows) != len(deep_rows):
            raise ValueError(f"权重 {w}>0 的辅助支路缺少等长记录行，拒绝静默跳过")
    fused = []
    for deep, *others in zip(deep_rows, *[rows for _, rows in aux]):
        probabilities = weights[0] * np.asarray(deep["probabilities"], dtype=np.float64)
        for (w, _), rows in zip(aux, others):
            probabilities = probabilities + w * np.asarray(rows["probabilities"], dtype=np.float64)
        fused.append({**deep, "probabilities": probabilities.tolist()})
    return fused


def _record_rows_by_key(rows, name):
    index = {}
    for row in rows:
        if row["record_key"] in index:
            raise ValueError(f"{name} 记录键重复：{row['record_key']}")
        index[row["record_key"]] = row
    return index


def _deep_provenance(info, folder):
    """校验并返回深度评估 info 的 last 重跑来源；字段缺失或 None 不允许绕过。"""
    provenance = info.get("provenance")
    if not isinstance(provenance, dict) or any(key not in provenance for key in PROVENANCE_KEYS):
        raise ValueError(f"深度评估缺少 last 重跑来源 provenance：{folder}")
    if provenance["checkpoint_kind"] != "last" or not isinstance(provenance["checkpoint"], str):
        raise ValueError(f"深度评估 provenance 不是带路径的 last：{folder}")
    if not isinstance(provenance["supervised_pool_units"], list):
        raise ValueError(f"深度评估监督池字段必须是列表（空池为 []）：{folder}")
    if not isinstance(provenance["source_split"], dict) or not isinstance(provenance["repr_config"], dict):
        raise ValueError(f"深度评估 provenance 缺少源折划分或表征配置：{folder}")
    scaler = provenance["scaler"]
    if (not isinstance(scaler, dict) or not isinstance(scaler.get("mean"), list)
            or not isinstance(scaler.get("std"), list)
            or len(scaler["mean"]) != 25 or len(scaler["std"]) != 25):
        raise ValueError(f"深度评估 provenance 缺少 25 维 scaler：{folder}")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"深度评估 scaler 必须有限且标准差为正：{folder}")
    return provenance


def _check_same_last(pu_provenance, cw_provenance, config, base_arm, split, expected_scaler):
    """CW/PU 深度评估必须来自同一个完成冻结轮数的 last（计划 §8）。"""
    epochs = config["parameters"]["epochs"]
    if pu_provenance["checkpoint"] != cw_provenance["checkpoint"]:
        raise ValueError("CW/PU 深度评估不是同一个 last checkpoint")
    for key in PROVENANCE_KEYS[1:]:
        if pu_provenance[key] != cw_provenance[key]:
            raise ValueError(f"CW/PU 深度评估 provenance.{key} 不一致")
    if pu_provenance["epoch"] != epochs or pu_provenance["last_epoch"] != epochs:
        raise ValueError(f"深度评估 provenance 必须完成冻结 {epochs} 轮的 last")
    if set(pu_provenance["supervised_pool_units"]) != set(_pool_units_for(config, base_arm)):
        raise ValueError("深度评估监督池与融合基座冻结口径不一致")
    source_split = pu_provenance["source_split"]
    if (set(source_split.get("train", [])) != set(split["train"])
            or set(source_split.get("val", [])) != set(split["val"])):
        raise ValueError("深度评估源折划分与当前 fold 冻结划分不一致")
    scaler = pu_provenance["scaler"]
    if not (np.allclose(scaler["mean"], expected_scaler["mean"], rtol=0, atol=1e-9)
            and np.allclose(scaler["std"], expected_scaler["std"], rtol=0, atol=1e-9)):
        raise ValueError("深度评估 scaler 不是本折源训练窗口 fit_scaler 的结果")


def _read_phys_ies_folder(eval_root, config):
    """读取 phys_ies 硬判单元（错误互补表用），不用 q_phys 冒充硬锚。"""
    folder = Path(eval_root) / "phys_ies"
    info = _read_json(folder / "eval_info.json")
    if (info.get("protocol") != "v8_stage3" or info.get("arm") != "phys_ies"
            or info.get("dataset") != "paderborn"
            or info.get("cache_version") != config["cache_version"]):
        raise ValueError(f"phys_ies 评估目录口径与融合不一致：{folder}")
    _check_pilot(info, "phys_ies", folder)
    path = folder / "unit_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"phys_ies 单元预测缺失，拒绝用 q_phys 替代：{path}")
    return read_predictions(path, unit=True)


def _error_complementarity(scopes, deep_units, phys_units):
    """冻结 all_primary/T_old/T_pair 上 deep 与 phys_ies 硬判的逐单元四格互补表。"""
    keys, _, (deep, phys) = _align_units([deep_units, phys_units], ("deep", "phys_ies"))
    table = {}
    for name in ("all_primary", "T_old", "T_pair"):
        wanted = sorted(scopes[name])
        if not wanted or not set(wanted) <= set(keys):
            raise ValueError(f"{name} 单元清单缺失或为空，无法统计错误互补")
        labels = [deep[key]["label"] for key in wanted]
        if any(label is None or type(label) is not int or label not in range(4) for label in labels):
            raise ValueError(f"{name} 含无真值单元，错误互补表只统计有标签单元")
        truth = np.asarray(labels)
        deep_pred = np.asarray([np.asarray(deep[key]["probabilities"], dtype=np.float64).argmax()
                                for key in wanted])
        phys_pred = np.asarray([np.asarray(phys[key]["probabilities"], dtype=np.float64).argmax()
                                for key in wanted])
        table[name] = {
            "unit_count": len(wanted),
            "cells": {"both_correct": int(((deep_pred == truth) & (phys_pred == truth)).sum()),
                      "only_deep_correct": int(((deep_pred == truth) & (phys_pred != truth)).sum()),
                      "only_phys_ies_correct": int(((deep_pred != truth) & (phys_pred == truth)).sum()),
                      "both_wrong": int(((deep_pred != truth) & (phys_pred != truth)).sum())},
            "P_ball": {"deep": float(np.mean(deep_pred == 1)),
                       "phys_ies": float(np.mean(phys_pred == 1))},
            "note": "逐单元硬判互补；phys_ies 为硬 one-hot（非 q_phys 软分），不含 KB 无真值单元"}
    return table


def _normal_evidence_distribution(normal_rows, quantile_method="linear"):
    """主PU/KB 记录级 Normal 证据描述与 a<=t 辅助判定；不借它调阈值、不改主分母。"""
    report = {}
    for scope, labeled in (("main_pu", True), ("kb", False)):
        rows = [row for row in normal_rows if (row["label"] is not None) == labeled]
        if not rows:
            report[scope] = {"record_count": 0}
            continue
        anomaly = np.asarray([row["anomaly_mean"] for row in rows], dtype=np.float64)
        window_fraction = np.asarray([row["normal_threshold_fraction"] for row in rows], dtype=np.float64)
        record_quantiles = np.asarray([row["anomaly_quantiles"] for row in rows], dtype=np.float64)
        if (not np.isfinite(anomaly).all() or not np.isfinite(window_fraction).all()
                or not np.isfinite(record_quantiles).all()):
            raise ValueError(f"Normal 证据字段包含非有限值：{scope}")
        report[scope] = {
            "record_count": len(rows),
            "anomaly_mean": float(anomaly.mean()),
            "anomaly_mean_quantiles": np.quantile(
                anomaly, [0.05, 0.5, 0.95], method=quantile_method).tolist(),
            "record_anomaly_quantile_mean": record_quantiles.mean(axis=0).tolist(),
            "mean_window_fraction_at_or_below_threshold": float(window_fraction.mean()),
            "records_all_windows_at_or_below_threshold_fraction":
                float(np.mean(window_fraction == 1.0))}
    report["note"] = ("记录级异常分均值/分位与窗口 a<=t 比例仅作描述；a<=t 是源验证经验分位下的"
                      "辅助正常判定，不是真实后验，不参与融合 argmax，也不改主分母")
    return report


def _read_eval_folder(folder, expect_arm, fold, seed, dataset, config):
    """读取同格深度评估目录；校验口径、last 来源与真实窗口区间后返回（记录行, 来源）。"""
    folder = Path(folder)
    info = _read_json(folder / "eval_info.json")
    if (info.get("protocol") != "v8_stage3" or info.get("arm") != expect_arm
            or info.get("fold") != fold or info.get("seed") != seed or info.get("dataset") != dataset
            or info.get("cache_version") != config["cache_version"]):
        raise ValueError(f"深度评估目录口径与融合格不一致：{folder}")
    _check_pilot(info, "深度评估", folder)
    for name in ("metrics.json", "predictions.csv", "unit_predictions.csv"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"深度评估报告不完整：{folder / name}")
    provenance = _deep_provenance(info, folder)
    rows = read_predictions(folder / "predictions.csv")
    for row in rows:
        if (type(row.get("start")) is not int or type(row.get("stop")) is not int
                or row["stop"] <= row["start"]):
            raise ValueError(f"深度评估记录必须带真实非空窗口区间：{folder}")
    return rows, provenance


def _run_fusion(config, args, execute):
    if args.arm not in FUSION_BASES:
        raise ValueError(f"{args.arm} 不是融合臂")
    if args.dataset not in (None, "paderborn"):
        raise ValueError("融合只在 PU 目标口径上评估")
    if args.fold is None or args.seed is None:
        raise ValueError("融合评估必须显式指定 --fold/--seed")
    _validate_fusion_config(config)
    base, fold, seed = FUSION_BASES[args.arm], args.fold, args.seed
    output_root = _output_dir(args.output or ROOT / "outputs/eval" / f"{args.arm}_f{fold}_s{seed}")
    outputs = {"full": output_root,
               "without_phys": output_root.parent / f"{output_root.name}_without_phys",
               "without_normal": output_root.parent / f"{output_root.name}_without_normal"}
    # 默认取正式产物根；测试通过 args 注入临时目录，不写真实 outputs。
    eval_root = Path(getattr(args, "deep_eval_root", None) or ROOT / "outputs/eval")
    train_root = Path(getattr(args, "normal_root", None) or ROOT / "outputs/train")
    normal_dir = train_root / f"normal_f{fold}"
    deep_dirs = {dataset: eval_root / f"{base}_f{fold}_s{seed}_{dataset}"
                 for dataset in ("cwru", "paderborn")}
    if not execute:
        print(json.dumps({"arm": args.arm, "fold": fold, "seed": seed, "execute": False,
                          "outputs": {k: str(v) for k, v in outputs.items()},
                          "dependencies": {
                              "normal_dir": str(normal_dir),
                              "normal_complete": all((normal_dir / n).is_file() for n in
                                                     ("model.joblib", "scaler.json", "config.json",
                                                      "split.json", "calibration.json")),
                              "deep_pu_eval": str(deep_dirs["paderborn"]),
                              "deep_cw_eval": str(deep_dirs["cwru"]),
                              "phys_ies_unit_predictions":
                                  str(eval_root / "phys_ies" / "unit_predictions.csv")}},
                     ensure_ascii=False))
        return
    import oneclass
    from features import feature_subset, fit_scaler, load_cache
    target = load_cache("paderborn", _project(config["cache_dir"]))
    source = load_cache("cwru", _project(config["cache_dir"]))
    selection = _read_json(_project(config["selection_path"]))
    # 复用同格深度末轮预测；两个数据集的评估目录必须来自同一个 last。
    deep_pu, pu_provenance = _read_eval_folder(deep_dirs["paderborn"], base, fold, seed, "paderborn", config)
    deep_cw, cw_provenance = _read_eval_folder(deep_dirs["cwru"], base, fold, seed, "cwru", config)
    split = _source_split(source, fold, config)
    expected_scaler = fit_scaler(feature_subset(source, sorted(split["train"]), labeled=False).phys25)
    _check_same_last(pu_provenance, cw_provenance, config, base, split, expected_scaler)
    paired = _paired_splits(config)
    phys_ies_units = _read_phys_ies_folder(eval_root, config)
    val_units = set(split["val"])
    deep_val = [row for row in deep_cw if row["unit_key"] in val_units]
    if not deep_val:
        raise ValueError("深度源验证记录为空，无法选择融合权重")
    # Normal 同折一份模型（seed 固定 0）；读入前核 config/split/run_kind。
    _check_auxiliary_model(config, normal_dir, "normal", fold, 0, source, selection)
    val_cache = feature_subset(source, sorted(val_units), labeled=False)
    normal_val = oneclass.predict_records(val_cache, normal_dir)
    normal_pu = oneclass.predict_records(target, normal_dir)
    q_phys_val = _q_phys_record_rows(val_cache, val_cache.records)
    q_phys_pu = _q_phys_record_rows(target, target.records)
    # 源验证三路证据对齐并选权重；缺一路记录直接报错，不做静默交集。
    weights_report, _ = _select_weights(aggregate_units(deep_val), aggregate_units(normal_val),
                                        aggregate_units(q_phys_val), config["fusion"]["weights"])
    weights = weights_report["chosen_weights"]
    # PU 记录级对齐：deep/normal/q_phys 键集合必须完全一致。
    deep_index = _record_rows_by_key(deep_pu, "深度")
    aux_maps = []
    for name, rows in (("normal", normal_pu), ("q_phys", q_phys_pu)):
        index = _record_rows_by_key(rows, name)
        if set(index) != set(deep_index):
            missing = sorted(set(deep_index) - set(index))[:3]
            extra = sorted(set(index) - set(deep_index))[:3]
            raise ValueError(f"{name} PU 记录与深度不一致：缺 {missing}，多 {extra}")
        for key, row in index.items():
            if row["unit_key"] != deep_index[key]["unit_key"] or row["label"] != deep_index[key]["label"]:
                raise ValueError(f"{name} 与深度记录真值不一致：{key}")
        aux_maps.append(index)
    keys = sorted(deep_index)
    deep_rows = [deep_index[key] for key in keys]
    aux_rows = [[m[key] for key in keys] for m in aux_maps]
    val_deep_index = _record_rows_by_key(deep_val, "深度源验证")
    val_aux_maps = []
    for name, rows in (("normal", normal_val), ("q_phys", q_phys_val)):
        index = _record_rows_by_key(rows, f"{name}源验证")
        if set(index) != set(val_deep_index):
            missing = sorted(set(val_deep_index) - set(index))[:3]
            extra = sorted(set(index) - set(val_deep_index))[:3]
            raise ValueError(f"{name} 源验证记录与深度不一致：缺 {missing}，多 {extra}")
        val_aux_maps.append(index)
    val_keys = sorted(val_deep_index)
    val_deep_rows = [val_deep_index[key] for key in val_keys]
    val_aux_rows = [[m[key] for key in val_keys] for m in val_aux_maps]
    versions = {"fusion_full": weights,
                "fusion_without_phys": _renormalize(weights, "without_phys"),
                "fusion_without_normal": _renormalize(weights, "without_normal")}
    # 三个版本：目标记录/单元级融合 + 源验证同权重融合（供各版本自己的 tau）。
    deep_val_units = sorted(aggregate_units(deep_val), key=lambda r: r["unit_key"])
    deep_pu_units = sorted(aggregate_units(deep_pu), key=lambda r: r["unit_key"])
    target_unit_keys = [row["unit_key"] for row in deep_pu_units]
    error_table = _error_complementarity(paired, deep_pu_units, phys_ies_units)
    fused_records, fused_units, target_prob, val_prob = {}, {}, {}, {}
    for name, w in versions.items():
        fused_records[name] = _fuse_records(w, deep_rows, *aux_rows)
        fused_units[name] = aggregate_units(fused_records[name])
        # 目标概率必须与 deep_pu_units 同序，保证 tau 报告的键一一对应。
        order = {row["unit_key"]: np.asarray(row["probabilities"], dtype=np.float64)
                 for row in fused_units[name]}
        if set(order) != set(target_unit_keys):
            raise ValueError(f"{name} 融合单元清单与深度不一致")
        target_prob[name] = np.asarray([order[key] for key in target_unit_keys])
        val_prob[name] = np.asarray(
            [row["probabilities"] for row in
             aggregate_units(_fuse_records(w, val_deep_rows, *val_aux_rows))], dtype=np.float64)
    # 原深度版本只引用不落盘，同样进入不确定报告。
    versions_prob = {name: (val_prob[name], target_prob[name]) for name in versions}
    versions_prob["deep_original"] = (
        np.asarray([row["probabilities"] for row in deep_val_units], dtype=np.float64),
        np.asarray([row["probabilities"] for row in deep_pu_units], dtype=np.float64))
    uncertainty = _uncertainty_report(versions_prob, deep_pu_units,
                                      quantile=config["fusion"]["uncertainty_quantile"])
    info_extra = {"fusion": {
        "base_arm": base, "chosen_weights": weights,
        "weight_names": config["fusion"]["weight_names"],
        "selection": {"scope": "source_val_unit_E1_macro_recall",
                      "unit_count": weights_report["unit_count"],
                      "chosen_source_val_macro_recall_E1": weights_report["chosen_source_val_macro_recall_E1"],
                      "tie_break": config["fusion"]["tie_break"], "candidates": weights_report["candidates"]},
        "versions": {name: {"weights": w, "equal_to_deep_original": bool(w == [1.0, 0.0, 0.0])}
                     for name, w in versions.items()},
        "normal_model_dir": str(normal_dir.resolve()),
        "deep_eval_dirs": {k: str(v.resolve()) for k, v in deep_dirs.items()},
        "uncertainty": uncertainty,
        "normal_evidence_distribution": _normal_evidence_distribution(normal_pu),
        "error_complementarity": error_table,
        "note": "权重只在源验证单元 E1 上选择，不优化 PU E2/P_ball；两路消融由完整权重"
                "置零归一化得到，不另调权；不确定标记不改主分母，Ball 仍可被 E1 选中"}}
    pool = _pool_units_for(config, args.arm)
    for name in ("fusion_full", "fusion_without_phys", "fusion_without_normal"):
        # 三版本必须落到不同 arm 名，汇总不得把消融当同格重复报告。
        version_arm = (args.arm if name == "fusion_full"
                       else f"{args.arm}_{name.removeprefix('fusion_')}")
        info, checked_pool = _stage3_info(config, version_arm, "paderborn", target, pool_units=pool,
                                          fold=fold, seed=seed,
                                          extra={**info_extra, "fusion_version": name,
                                                 "weights_used": versions[name]})
        save_reports(fused_records[name], fused_units[name],
                     outputs["full" if name == "fusion_full" else name.replace("fusion_", "")],
                     pool_units=checked_pool, eval_info=info)
    # 权重随 calibration.json 落盘在完整融合目录。
    with (outputs["full"] / "calibration.json").open("x", encoding="utf-8") as handle:
        json.dump({"chosen_weights": weights, "tie_break": config["fusion"]["tie_break"],
                   "selection_scope": "source_val_unit_E1_macro_recall",
                   "source_val_unit_count": weights_report["unit_count"],
                   "candidates": weights_report["candidates"]},
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"融合评估已落盘：{outputs['full']}（含两路消融），权重 {weights}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="阶段三只读评估（深度/浅层/融合），默认 dry-run")
    parser.add_argument("--config", default="configs/stage3.json")
    parser.add_argument("--arm", default=None,
                        choices=("joint", "joint_logspec", "mixup4src", "coral4src", "adabn",
                                 "ce_K19", "ce_K19_gg", "ce_K116", "ce_K19_logspec",
                                 "svm_phys", "fusion_joint", "fusion_ce_K19"),
                        help="深度评估可省略，默认取 checkpoint 的臂")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-dir")
    parser.add_argument("--dataset", choices=("cwru", "paderborn"), default=None)
    parser.add_argument("--fold", type=int, default=None,
                        help="深度评估可省略，默认取 checkpoint 的折")
    parser.add_argument("--seed", type=int, default=None,
                        help="深度评估可省略，默认取 checkpoint 的种子")
    parser.add_argument("--output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from oneclass import load_config
    config = load_config(args.config)
    if ((args.fold is not None and args.fold not in config["folds"])
            or (args.seed is not None and args.seed not in config["seeds"])):
        raise ValueError("fold/seed 不在冻结矩阵中")
    runner = (_run_fusion if args.arm in FUSION_BASES
              else _run_shallow if args.arm == "svm_phys" else _run_deep)
    runner(config, args, execute=args.execute)


if __name__ == "__main__":
    main()
