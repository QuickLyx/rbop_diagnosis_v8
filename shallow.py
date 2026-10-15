# -*- coding: utf-8 -*-
"""旧 K19 的固定物理 SVM、显式源单元留出校准与探索性粗糙集。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import time
import tracemalloc

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from oneclass import dump_json, load_config, new_output, peak_memory_bytes, project_path, source_split


def allocate_quota(lengths, budget=64):
    """按已排序记录等分配额，短记录余量按同序轮转，不重复取窗。"""
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any() or budget <= 0:
        raise ValueError("每条记录窗口数及预算必须为正")
    total = min(int(lengths.sum()), int(budget))
    quotas = np.minimum(lengths, total // len(lengths) + (np.arange(len(lengths)) < total % len(lengths)))
    remaining = total - int(quotas.sum())
    while remaining:
        for i in range(len(quotas)):
            if quotas[i] < lengths[i]:
                quotas[i] += 1
                remaining -= 1
                if not remaining:
                    break
    return quotas


def sample_units(cache, unit_keys, budget=64):
    """先单元、再记录键排序，固定分位中心索引抽样。"""
    units = set(unit_keys)
    if units - {r["unit_key"] for r in cache.records}:
        raise ValueError("配额单元不在缓存中")
    indices, groups, labels, domains, metadata = [], [], [], [], []
    for unit in sorted(units):
        rows = sorted((r for r in cache.records if r["unit_key"] == unit), key=lambda r: r["record_key"])
        if any(r["label"] is None for r in rows) or len({r["label"] for r in rows}) != 1:
            raise ValueError("SVM 拟合单元缺真值或标签不一致")
        if rows[0]["dataset"] == "cwru" and Counter(r["channel"] for r in rows) != {"DE": 1, "FE": 1}:
            raise ValueError("CWRU 配额必须含 DE/FE 两通道")
        lengths = [r["stop"] - r["start"] for r in rows]
        quotas = allocate_quota(lengths, budget)
        for row, n, q in zip(rows, lengths, quotas):
            chosen = np.floor((np.arange(q) + 0.5) * n / q).astype(int) if q else np.array([], dtype=int)
            if len(np.unique(chosen)) != q or (chosen >= n).any():
                raise AssertionError("配额抽样发生重复或越界")
            indices.extend((row["start"] + chosen).tolist())
            labels.extend([row["label"]] * int(q))
            groups.extend([f"{row['dataset']}/{unit}"] * int(q))
            domains.extend([row["dataset"]] * int(q))
            metadata.append({"unit_key": unit, "record_key": row["record_key"], "available": n,
                             "quota": int(q), "local_indices": chosen.tolist()})
    return {"indices": np.asarray(indices, dtype=int), "groups": np.asarray(groups),
            "y": np.asarray(labels, dtype=int), "domains": np.asarray(domains), "quotas": metadata}


def training_weights(y, groups, domains):
    """源/目标各半，域内类、类内单元、单元内所选窗分别等权。"""
    y, groups, domains = np.asarray(y), np.asarray(groups), np.asarray(domains)
    if not (len(y) == len(groups) == len(domains)) or set(domains) != {"cwru", "paderborn"}:
        raise ValueError("SVM 拟合必须同时含源/旧池两域")
    weights = np.empty(len(y), dtype=np.float64)
    for domain in sorted(set(domains)):
        labels = sorted(set(y[domains == domain]))
        for label in labels:
            mask = (domains == domain) & (y == label)
            units = sorted(set(groups[mask]))
            for unit in units:
                inside = mask & (groups == unit)
                weights[inside] = 1 / (2 * len(labels) * len(units) * inside.sum())
    return weights / weights.mean()


def candidate_score(y, predicted, groups, domains):
    """内部选择分数：每单元窗口正确率→类均值→源/目标各半。"""
    parts = []
    for domain in ("cwru", "paderborn"):
        recalls = []
        for label in sorted(set(y[domains == domain])):
            units = sorted(set(groups[(domains == domain) & (y == label)]))
            recalls.append(np.mean([np.mean(predicted[groups == unit] == y[groups == unit]) for unit in units]))
        if not recalls:
            raise ValueError("内部验证缺失一个域")
        parts.append(float(np.mean(recalls)))
    return float(np.mean(parts))


def calibration_weights(feature):
    """源验证类→文件→通道→窗口等权，不拿窗口随机 CV 校准。"""
    rows = feature.records
    labels = {r["label"] for r in rows}
    if labels != {0, 1, 2, 3}:
        raise ValueError("独立源验证校准集必须含四类")
    weights = np.empty(len(feature.phys25), dtype=np.float64)
    for label in sorted(labels):
        units = {r["unit_key"] for r in rows if r["label"] == label}
        for unit in sorted(units):
            channels = [r for r in rows if r["unit_key"] == unit]
            for row in channels:
                n = row["stop"] - row["start"]
                weights[row["start"]:row["stop"]] = 1 / (4 * len(units) * len(channels) * n)
    return weights / weights.mean()


def class_probabilities(model, x):
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    if len(classes) != 4 or set(classes.tolist()) != {0, 1, 2, 3}:
        raise ValueError("校准器必须包含四个固定类别")
    result = np.empty((len(x), 4), dtype=np.float64)
    result[:, classes.astype(int)] = probabilities
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1):
        raise ValueError("SVM 输出概率非有限或未归一")
    return result


def fit_shallow(source_cache, target_cache, selection, fold, seed, output, config):
    from features import feature_subset, fit_scaler, scale_phys
    if seed not in config["seeds"]:
        raise ValueError("seed 不在冻结矩阵中")
    params = config["svm"]
    if (params["C_grid"] != [0.1, 1, 10] or params["gamma_grid"] != [0.01, 0.04, 0.1]
            or params["windows_per_unit"] != 64 or params["internal_folds"] != 3
            or params["kernel"] != "rbf" or params["probability"] is not False
            or params["decision_function_shape"] != "ovr" or params["tol"] != 1e-3
            or params["cache_size"] != 1024 or params["max_iter"] != -1):
        raise ValueError("SVM 配方偏离固定计划")
    split = source_split(source_cache, fold, config)
    pool = selection["budgets"]["K19"]["pool"]
    all_units = set(selection["all"])
    test = selection["budgets"]["K19"]["test"]
    if (len(all_units) != 116 or len(selection["all"]) != 116 or len(pool) != 19
            or len(set(pool)) != 19 or len(test) != 97 or len(set(test)) != 97
            or set(pool) & set(test) or set(pool) | set(test) != all_units):
        raise ValueError("SVM 旧 K19 池与自然留出不完整或泄漏")
    src = sample_units(source_cache, split["train"])
    tgt = sample_units(target_cache, pool)
    raw = np.concatenate((np.asarray(source_cache.phys25)[src["indices"]], np.asarray(target_cache.phys25)[tgt["indices"]]))
    y = np.concatenate((src["y"], tgt["y"]))
    groups = np.concatenate((src["groups"], tgt["groups"]))
    domains = np.concatenate((src["domains"], tgt["domains"]))
    if set(y) != {0, 1, 2, 3}:
        raise ValueError("SVM 外层拟合缺四类")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    internal = []
    for train_ids, val_ids in cv.split(raw, y, groups):
        if set(groups[train_ids]) & set(groups[val_ids]) or set(y[train_ids]) != {0, 1, 2, 3}:
            raise ValueError("内部折有单元泄漏或缺四类，不自动改种子")
        source_units = sorted(g.removeprefix("cwru/") for g in set(groups[train_ids]) if g.startswith("cwru/"))
        if not source_units:
            raise ValueError("内部训练折没有源单元")
        scaler = fit_scaler(feature_subset(source_cache, source_units).phys25)
        internal.append((train_ids, val_ids, scaler))
    output = new_output(output)
    dump_json(output / "split.json", {"source": split, "target": {"pool": pool, "test": test},
        "internal": [{"train_groups": sorted(set(groups[a])), "val_groups": sorted(set(groups[b])),
                      "source_scaler_units": sorted(g for g in set(groups[a]) if g.startswith("cwru/"))}
                     for a, b, _ in internal]})
    dump_json(output / "quotas.json", src["quotas"] + tgt["quotas"])
    start = time.perf_counter()
    tracemalloc.start()
    grid, best = [], None
    try:
        for c in params["C_grid"]:
            for gamma in params["gamma_grid"]:
                predicted = np.full(len(y), -1, dtype=int)
                fold_costs = []
                for train_ids, val_ids, scaler in internal:
                    stamp = time.perf_counter()
                    svc = SVC(C=c, gamma=gamma, kernel="rbf", probability=False,
                              decision_function_shape="ovr", tol=1e-3, cache_size=1024, max_iter=-1)
                    svc.fit(scale_phys(raw[train_ids], scaler), y[train_ids],
                            sample_weight=training_weights(y[train_ids], groups[train_ids], domains[train_ids]))
                    if svc.fit_status_ != 0:
                        raise RuntimeError("SVM 内部拟合未成功收敛")
                    predicted[val_ids] = svc.predict(scale_phys(raw[val_ids], scaler))
                    fold_costs.append({"elapsed_seconds": time.perf_counter()-stamp,
                                       "support_vector_count": int(svc.n_support_.sum())})
                if (predicted < 0).any():
                    raise AssertionError("内部池外预测没有覆盖所有配额窗口")
                score = candidate_score(y, predicted, groups, domains)
                grid.append({"C": c, "gamma": gamma, "score": score, "folds": fold_costs})
                key = (-score, c, gamma)
                if best is None or key < best[0]:
                    best = (key, c, gamma)
        train_source = feature_subset(source_cache, split["train"])
        val_source = feature_subset(source_cache, split["val"])
        scaler = fit_scaler(train_source.phys25)
        svc = SVC(C=best[1], gamma=best[2], kernel="rbf", probability=False,
                  decision_function_shape="ovr", tol=1e-3, cache_size=1024, max_iter=-1)
        svc.fit(scale_phys(raw, scaler), y, sample_weight=training_weights(y, groups, domains))
        if svc.fit_status_ != 0:
            raise RuntimeError("最终 SVM 未成功收敛")
        frozen = FrozenEstimator(svc)
        calibrated = CalibratedClassifierCV(frozen, method="sigmoid")
        support_before = svc.support_.copy()
        calibrated.fit(scale_phys(val_source.phys25, scaler), np.asarray(val_source.y),
                       sample_weight=calibration_weights(val_source))
        if not np.array_equal(svc.support_, support_before):
            raise AssertionError("FrozenEstimator 校准错误地重新拟合了 SVC")
        _, peak = tracemalloc.get_traced_memory()
        joblib.dump(calibrated, output / "model.joblib")
        dump_json(output / "scaler.json", scaler)
        dump_json(output / "config.json", {**config, "arm": "svm_phys", "fold": fold, "seed": seed,
            "run_kind": "pilot" if output.resolve().is_relative_to((project_path("outputs/pilot")).resolve()) else "formal"})
        dump_json(output / "calibration.json", {"method": "sigmoid", "estimator": "FrozenEstimator",
                  "units": split["val"], "windows": len(val_source.phys25),
                  "note": "源验证只用于独立单元留出校准；校准后分数不再是独立源泛化指标"})
        dump_json(output / "fit_info.json", {"grid": grid, "C": best[1], "gamma": best[2],
                  "fit_count": len(grid)*len(internal)+1, "calibration_fit_count": 1,
                  "elapsed_seconds": time.perf_counter()-start, "python_traced_peak_bytes": peak,
                  "process_peak_memory_bytes": peak_memory_bytes(),
                  "memory_note": "OS进程历史峰值并非算法独占增量；tracemalloc不保证涵盖libsvm原生缓存",
                  "support_vector_count": int(svc.n_support_.sum()), "sklearn_version": sklearn.__version__})
    finally:
        tracemalloc.stop()
    return output


def predict_records(cache, model_dir):
    from features import scale_phys
    from evaluate import aggregate_records
    folder = project_path(model_dir)
    if any(not (folder / name).is_file() for name in
           ("model.joblib", "scaler.json", "config.json", "split.json", "calibration.json", "fit_info.json", "quotas.json")):
        raise FileNotFoundError(f"SVM 产物不完整：{folder}")
    metadata = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "v8_stage3" or metadata.get("arm") != "svm_phys":
        raise ValueError("不是阶段三 svm_phys 模型")
    model = joblib.load(folder / "model.joblib")
    scaler = json.loads((folder / "scaler.json").read_text(encoding="utf-8"))
    probabilities = np.empty((len(cache.phys25), 4), dtype=np.float64)
    for start in range(0, len(probabilities), 4096):
        probabilities[start:start+4096] = class_probabilities(model, scale_phys(cache.phys25[start:start+4096], scaler))
    return aggregate_records(cache.records, probabilities)


def rough_dependency(discrete, labels):
    groups = {}
    for signature, label in zip(map(tuple, discrete), labels):
        groups.setdefault(signature, []).append(int(label))
    pure = [rows for rows in groups.values() if len(set(rows)) == 1]
    return sum(map(len, pure)) / len(labels), len(groups), len(pure)


def rough_set_analysis(cache, pool, scaler):
    if len(pool) != 19 or len(set(pool)) != 19:
        raise ValueError("RS 必须仅使用冻结旧 K19")
    from features import scale_phys
    matrix, labels = [], []
    for unit in pool:
        rows = [r for r in cache.records if r["unit_key"] == unit]
        if not rows or len({r["label"] for r in rows}) != 1 or rows[0]["label"] is None:
            raise ValueError("RS 单元缺失或真值不一致")
        matrix.append(np.mean([scale_phys(cache.phys25[r["start"]:r["stop"]], scaler).mean(axis=0) for r in rows], axis=0))
        labels.append(rows[0]["label"])
    matrix = np.asarray(matrix)
    boundaries, columns = [], []
    for col in matrix.T:
        edges = np.unique(np.quantile(col, [0.25, 0.5, 0.75], method="linear"))
        if np.all(col == col[0]):
            edges = np.empty(0)
        boundaries.append(edges.tolist())
        columns.append(np.searchsorted(edges, col, side="right"))
    discrete = np.column_stack(columns)
    gamma, signatures, pure = rough_dependency(discrete, labels)
    importance = []
    for index in range(25):
        removed, _, _ = rough_dependency(np.delete(discrete, index, axis=1), labels)
        importance.append({"dimension": index, "importance": gamma-removed, "gamma_without": removed})
    return {"pool": list(pool), "support": dict(Counter(labels)), "gamma": gamma,
            "unique_signatures": signatures, "pure_equivalence_classes": pure,
            "boundaries": boundaries, "discrete": discrete.tolist(), "importance": importance,
            "note": "仅旧19单元的探索性正域分析，不代表显著性/因果性，不反馈主训练"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定物理 SVM 与探索性 RS；默认不拟合")
    parser.add_argument("--config", default="configs/stage3.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--rs", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.fold not in config["folds"] or args.seed not in config["seeds"]:
        raise ValueError("fold/seed 不在固定矩阵中")
    if not args.execute:
        print(json.dumps({"arm": "rs" if args.rs else "svm_phys", "fold": args.fold,
                          "seed": args.seed, "fit": False, "parameters": config["svm"]}, ensure_ascii=False))
        return
    from features import load_cache, feature_subset, fit_scaler
    config_cache = project_path(config["cache_dir"])
    source, target = load_cache("cwru", config_cache), load_cache("paderborn", config_cache)
    selection = json.loads(project_path(config["selection_path"]).read_text(encoding="utf-8"))
    if args.rs:
        split = source_split(source, args.fold, config)
        scaler = fit_scaler(feature_subset(source, split["train"]).phys25)
        result = rough_set_analysis(target, selection["budgets"]["K19"]["pool"], scaler)
        result["source_scaler_fold"] = args.fold
        output = new_output(args.output or "outputs/eval/rs")
        dump_json(output / "analysis.json", result)
    else:
        fit_shallow(source, target, selection, args.fold, args.seed,
                    args.output or f"outputs/train/svm_phys_f{args.fold}_s{args.seed}", config)


if __name__ == "__main__":
    main()
