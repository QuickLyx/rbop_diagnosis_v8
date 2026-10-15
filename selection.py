# -*- coding: utf-8 -*-
"""v8 选标：旧池只读复制、正则化 Gath-Geva 新池与配对留出（计划 §4）。

选标只用 116 主口径单元的 logspec512 单元均值，不读标签/严重度/损伤来源；
标签只在池落定后作覆盖诊断，不回流选标规则。旧池来自 v7 冻结 JSON，
按原始字节只读复制，不重新选标。
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from bearing_facts import PRIMARY_BEARINGS

ROOT = Path(__file__).resolve().parent
PROTOCOL = "v8_stage3"

#: 冻结 GG 配方（计划 §4）；configs/stage3.json 的 gg 块必须逐项一致，偏离即报错。
GG_DEFAULTS = {"m": 2, "seed": 0, "n_init": 25, "max_iter": 200, "tol": 1e-5,
               "covariance_weight": 0.9, "identity_weight": 0.1,
               "minimum_center_distance_squared": 1e-12, "pool_size": 19}
K_SEARCH = tuple(range(6, 20))


class _InvalidK(Exception):
    """单个 k 的 GG 判无效原因。"""


def unit_level_features(cache):
    """116 主口径单元内全部 logspec512 窗口等权均值（v7 纯函数本地保留）。"""
    groups = {}
    for record in cache.records:
        if record["bearing_id"] not in PRIMARY_BEARINGS:
            continue
        groups.setdefault(record["unit_key"], []).append(record)
    keys = sorted(groups)
    if len(keys) != 116:
        raise ValueError(f"主口径必须恰有 116 个单元，实际为 {len(keys)}")
    matrix = []
    for key in keys:
        records = groups[key]
        total = sum(record["stop"] - record["start"] for record in records)
        if total <= 0:
            raise ValueError(f"单元没有可用窗口：{key}")
        summed = sum(np.asarray(cache.X[r["start"]:r["stop"], 0], dtype=np.float64).sum(axis=0)
                     for r in records)
        matrix.append(summed / total)
    return keys, np.stack(matrix)


def validate_selection_payload(payload):
    """选标 JSON 的冻结完整性校验：budgets.K19/K116 与 116 全集不重不漏。"""
    if not isinstance(payload, dict) or "budgets" not in payload or "all" not in payload:
        raise ValueError("选标文件缺少 budgets/all 字段")
    all_units = payload["all"]
    from bearing_facts import CONDITIONS
    expected = {f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS for condition in CONDITIONS}
    if len(all_units) != 116 or len(set(all_units)) != 116 or set(all_units) != expected:
        raise ValueError("选标全集必须是 116 个不同单元")
    for name, size in (("K19", 19), ("K116", 116)):
        split = payload["budgets"].get(name)
        if not isinstance(split, dict) or "pool" not in split or "test" not in split:
            raise ValueError(f"选标文件缺少 budgets.{name}.pool/test")
        pool, test = split["pool"], split["test"]
        if (len(pool) != size or len(set(pool)) != size or len(test) != 116 - size
                or len(set(test)) != len(test) or set(pool) & set(test) or set(pool) | set(test) != set(all_units)):
            raise ValueError(f"选标 budgets.{name} 池/测试不满足不重不漏补集")
    return payload


def load_selection(path):
    """读取冻结选标 JSON；保持 budgets.K19.pool/test、budgets.K116.pool/test 字段。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"选标文件缺失：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_selection_payload(payload)


def import_old_selection(source, destination):
    """旧池只读复制：字节级复制到 v8 后用 load_selection 校验，不重新选标。"""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"旧选标文件缺失：{source}")
    payload = load_selection(source)
    if payload.get("protocol") != "v7_stage2" or payload.get("k_selected") != 19:
        raise ValueError("旧池必须是 v7_stage2 且 k_selected=19 的冻结选标")
    if destination.exists():
        if destination.read_bytes() == source.read_bytes():
            return payload
        raise FileExistsError(f"目标旧池文件已存在且内容不同，不覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.read_bytes() != source.read_bytes():
        raise ValueError("旧池复制后字节不一致，拒绝使用")
    return payload


def _gg_initialization(z, k, cfg):
    """固定 KMeans(n_init=25, seed=0) 硬分配 → 本簇 0.99、其余均分 0.01/(k-1)。"""
    fitted = KMeans(n_clusters=k, n_init=cfg["n_init"], random_state=cfg["seed"])
    labels = fitted.fit_predict(z)
    u = np.full((len(z), k), 0.01 / (k - 1))
    u[np.arange(len(z)), labels] = 0.99
    return u


def _weighted_moments(z, u, cfg):
    """加权中心、正则协方差 Creg=0.9C+0.1I 与混合权重；空簇权重和为 0 判无效。"""
    weights = u ** cfg["m"]
    totals = weights.sum(axis=0)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise _InvalidK("empty_cluster")
    dim = z.shape[1]
    centers = (weights.T @ z) / totals[:, None]
    covs = np.empty((u.shape[1], dim, dim))
    for cluster in range(u.shape[1]):
        diff = z - centers[cluster]
        covs[cluster] = (weights[:, cluster][:, None] * diff).T @ diff / totals[cluster]
        covs[cluster] *= cfg["covariance_weight"]
        covs[cluster] += cfg["identity_weight"] * np.eye(dim)
    pis = u.mean(axis=0)
    if (pis <= 0).any() or not np.isfinite(pis).all():
        raise _InvalidK("empty_cluster")
    return centers, covs, pis


def _gath_geva_single(z, k, cfg):
    """单 k 的 GG 迭代；返回最终隶属度。无效原因抛 _InvalidK。"""
    u = _gg_initialization(z, k, cfg)
    for _iteration in range(cfg["max_iter"]):
        centers, covs, pis = _weighted_moments(z, u, cfg)
        # logdet 走 Cholesky 对角，不直接求行列式；Mahalanobis 用三角求解，不求逆。
        chol = np.linalg.cholesky(covs)
        logdet = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        log_pi = np.log(pis)
        ell = np.empty((len(z), k))
        for cluster in range(k):
            diff = z - centers[cluster]
            solved = np.linalg.solve(chol[cluster], diff.T)
            mahal2 = (solved ** 2).sum(axis=0)
            ell[:, cluster] = 0.5 * logdet[cluster] - log_pi[cluster] + 0.5 * mahal2
        if not np.isfinite(ell).all():
            raise _InvalidK("non_finite")
        shifted = ell - ell.min(axis=1, keepdims=True)
        new_u = np.exp(-shifted / (cfg["m"] - 1))
        new_u /= new_u.sum(axis=1, keepdims=True)
        if not np.isfinite(new_u).all():
            raise _InvalidK("non_finite")
        converged = np.max(np.abs(new_u - u)) <= cfg["tol"]
        u = new_u
        if converged:
            return u
    raise _InvalidK("max_iter")


def _xb_score(z, u, centers, m, pairwise_min_squared):
    """XB = Σ u^m ||z-v||² / (n · min_{a≠b} ||v_a-v_b||²)；分母过小已在判据中排除。"""
    diff = z[:, None, :] - centers[None, :, :]
    distances = (diff ** 2).sum(axis=2)
    return float((u ** m * distances).sum() / (len(z) * pairwise_min_squared))


def gath_geva(z, k, cfg):
    """单 k 的 GG 拟合：无效 k 返回 {"valid": False, "reason": ...}，不重试。"""
    try:
        u = _gath_geva_single(z, k, cfg)
        centers, covs, pis = _weighted_moments(z, u, cfg)
    except (_InvalidK, np.linalg.LinAlgError, FloatingPointError) as error:
        return {"valid": False, "reason": str(error)}
    if not np.isfinite(centers).all() or not np.isfinite(u).all() or not np.isfinite(pis).all():
        return {"valid": False, "reason": "non_finite"}
    if len(np.unique(u.argmax(axis=1))) != k:
        # 硬分配空簇按计划判无效；同分 argmax 取较小簇号。
        return {"valid": False, "reason": "empty_cluster"}
    diff = centers[:, None, :] - centers[None, :, :]
    pairwise = (diff ** 2).sum(axis=2)
    np.fill_diagonal(pairwise, np.inf)
    min_distance = float(pairwise.min())
    if min_distance <= cfg["minimum_center_distance_squared"]:
        return {"valid": False, "reason": "center_distance"}
    xb = _xb_score(z, u, centers, cfg["m"], min_distance)
    if not np.isfinite(xb):
        return {"valid": False, "reason": "non_finite"}
    return {"valid": True, "u": u, "centers": centers, "covs": covs, "pis": pis,
            "xb": xb, "min_center_distance_squared": min_distance}


def select_gg(keys, matrix, cfg=None):
    """正则化 Gath-Geva 选标：k∈6..19 固定初始化，XB 最小（精确同分取较小 k）。"""
    merged = {**GG_DEFAULTS, **(cfg or {})}
    if merged != GG_DEFAULTS:
        raise ValueError("GG参数必须采用固定阶段三配方")
    if len(set(keys)) != len(keys) or len(keys) != 116 or matrix.shape != (116, 256):
        raise ValueError("选标键必须唯一，特征形状必须为 (单元数,256)")
    if not np.isfinite(matrix).all():
        raise ValueError("选标特征包含非有限值")
    z = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-12)
    curve, fits, invalid = {}, {}, {}
    elapsed = {}
    with threadpool_limits(limits=1):
        for k in K_SEARCH:
            started = time.perf_counter()
            result = gath_geva(z, k, merged)
            elapsed[str(k)] = time.perf_counter() - started
            if result["valid"]:
                curve[k] = result["xb"]
                fits[k] = result
            else:
                invalid[k] = result["reason"]
    if not curve:
        raise ValueError(f"全部 k 的 GG 均无效：{invalid}；按计划停在选标阶段，不改参数重试")
    selected = min(curve, key=lambda k: (curve[k], k))
    fit = fits[selected]
    u, centers, covs = fit["u"], fit["centers"], fit["covs"]
    hard = u.argmax(axis=1)  # 同分取较小簇号
    members = {}
    for cluster in range(selected):
        rows = np.flatnonzero(hard == cluster)
        diff = z[rows] - centers[cluster]
        solved = np.linalg.solve(np.linalg.cholesky(covs[cluster]), diff.T)
        mahal2 = (solved ** 2).sum(axis=0)
        ranked = sorted(zip(rows.tolist(), mahal2.tolist()),
                        key=lambda pair: (pair[1], keys[pair[0]]))
        members[cluster] = [index for index, _ in ranked]
    # 沿 v7 轮转：非单例簇在前、再按簇号；先各簇一代表，再逐深度轮转。
    order = sorted(members, key=lambda c: (len(members[c]) == 1, c))
    ranked_units = []
    for depth in range(max(map(len, members.values()))):
        for cluster in order:
            if depth < len(members[cluster]):
                ranked_units.append(keys[members[cluster][depth]])
    assert len(ranked_units) == len(keys) and len(set(ranked_units)) == len(keys)
    payload = {
        "protocol": PROTOCOL, "rule": "regularized_gath_geva",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "k_elapsed_seconds": elapsed, "elapsed_seconds": sum(elapsed.values()),
        "k_selected": selected,
        "xb_curve": {str(k): curve[k] for k in sorted(curve)},
        "valid_k": sorted(curve),
        "invalid_k": {str(k): invalid[k] for k in sorted(invalid)},
        "cluster_sizes": {str(c): len(members[c]) for c in members},
        "cluster_labels": {key: int(hard[i]) for i, key in enumerate(keys)},
        "medoids": [keys[members[c][0]] for c in order],
        "ranked_units": ranked_units, "all": keys,
        "selection_metadata_used": False,
        "label_usage": "选标不读标签/严重度/损伤来源；标签仅用于池落定后的覆盖诊断",
        "gg_config": {**merged, "k_search": list(K_SEARCH)},
        "min_center_distance_squared": fit["min_center_distance_squared"],
    }
    payload["budgets"] = gg_budget_splits(payload)
    return payload


def gg_budget_splits(payload, budgets=(GG_DEFAULTS["pool_size"], 116)):
    """GG 冻结序列的前缀预算：池与测试不重不漏覆盖全部 116 单元。"""
    all_keys, ranked = payload["all"], payload["ranked_units"]
    if len(set(ranked)) != len(all_keys) or set(ranked) != set(all_keys):
        raise ValueError("GG 代表序列必须不重不漏地覆盖全部单元")
    result = {}
    for size in budgets:
        if size not in (GG_DEFAULTS["pool_size"], len(all_keys)):
            raise ValueError(f"GG 预算只支持 K{GG_DEFAULTS['pool_size']} 与 K116：{size}")
        pool = sorted(ranked[:size])
        test = sorted(set(all_keys) - set(pool))
        assert not set(pool) & set(test) and set(pool) | set(test) == set(all_keys)
        result[f"K{size}"] = {"pool": pool, "test": test,
                              "label_unit_count": len(pool), "heldout_unit_count": len(test)}
    return result


def diagnostics(cache, splits):
    """池落定后才读真值，仅记录覆盖情况，不回流到选标规则。"""
    by_unit = {r["unit_key"]: r for r in cache.records if r["bearing_id"] in PRIMARY_BEARINGS}
    result = {}
    for name, split in splits.items():
        pool, test = split["pool"], split["test"]
        seen = {by_unit[key]["bearing_id"] for key in pool}
        result[name] = {
            "pool_class_counts": dict(Counter(str(by_unit[k]["label"]) for k in pool)),
            "test_class_counts": dict(Counter(str(by_unit[k]["label"]) for k in test)),
            "pool_origin_counts": dict(Counter(by_unit[k]["origin"] for k in pool)),
            "pool_condition_counts": dict(Counter(by_unit[k]["condition"] for k in pool)),
            "seen_bearings": sorted(seen),
            "unseen_bearings": sorted(set(PRIMARY_BEARINGS) - seen),
            "unseen_test_units": [k for k in test if by_unit[k]["bearing_id"] not in seen],
            "missing_pool_classes": sorted({0, 2, 3} - {by_unit[k]["label"] for k in pool}),
        }
    return result


def paired_splits(selection_old, selection_gg):
    """计划 §4 配对清单：自然留出、共同留出与共同未见轴承；不读任何标签。"""
    validate_selection_payload(selection_old)
    validate_selection_payload(selection_gg)
    u = sorted(set(selection_old["all"]))
    if len(u) != 116 or set(selection_gg["all"]) != set(u):
        raise ValueError("新旧选标全集必须同为 116 个单元")
    p_old = sorted(set(selection_old["budgets"]["K19"]["pool"]))
    p_gg = sorted(set(selection_gg["budgets"]["K19"]["pool"]))
    u_set = set(u)
    if set(p_old) - u_set or set(p_gg) - u_set or len(p_old) != 19 or len(p_gg) != 19:
        raise ValueError("两个监督池必须是全集内的 19 个单元")
    t_old = sorted(u_set - set(p_old))
    t_gg = sorted(u_set - set(p_gg))
    t_pair = sorted(u_set - (set(p_old) | set(p_gg)))
    seen = {unit.split("/", 1)[0] for unit in p_old} | {unit.split("/", 1)[0] for unit in p_gg}
    result = {"U": u, "P_old": p_old, "P_gg": p_gg, "T_old": t_old, "T_gg": t_gg,
              "T_pair": t_pair,
              "common_unseen_bearings": sorted(set(PRIMARY_BEARINGS) - seen),
              "common_unseen_units": [unit for unit in t_pair
                                      if unit.split("/", 1)[0] not in seen]}
    if set(result["common_unseen_units"]) - set(t_pair):
        raise ValueError("共同未见单元必须落在 T_pair 内")
    return result


def _gg_config(config):
    """校验 configs 的 gg 块与冻结配方逐项一致。"""
    cfg = dict(config.get("gg") or {})
    for key, value in GG_DEFAULTS.items():
        if cfg.get(key) != value:
            raise ValueError(f"configs 的 gg.{key}={cfg.get(key)!r} 偏离冻结值 {value!r}")
    if cfg.get("k_search") != list(K_SEARCH):
        raise ValueError("gg.k_search 必须固定为 6..19")
    return GG_DEFAULTS


def _project(value):
    """配置里的相对路径相对 v8 根解析。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path, payload):
    """新产物只写一次；非空目录/已有文件不覆盖（配对文件确定性一致时允许复用）。"""
    path = Path(path)
    if path.exists():
        same = json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(payload))
        if same:
            return False
        raise FileExistsError(f"产物已存在且内容不同，不覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return True


def run_selection(config, execute=False):
    """选标链：旧池字节复制 → GG 新池（K19/K116 预算）→ 配对清单。

    dry-run 只检查配置与输入存在性，不读缓存、不计算、不写文件。
    """
    gg_cfg = _gg_config(config)
    old_path = _project(config["old_selection_path"])
    new_path = _project(config["selection_gg_path"])
    old_copy_path = _project(config["selection_path"])
    paired_path = _project(config["paired_splits_path"])
    if not execute:
        status = {"old_selection": old_path.is_file(), "old_copy_target_exists": old_copy_path.exists(),
                  "gg_output_exists": new_path.exists(), "paired_exists": paired_path.exists(),
                  "cache_dir": str(_project(config["cache_dir"]))}
        return {"mode": "dry_run", "gg_config": gg_cfg, "status": status}
    for output in (old_copy_path, new_path, paired_path):
        if not output.resolve().is_relative_to((ROOT / "outputs").resolve()):
            raise ValueError("选标产物只允许写入v8/outputs")
    if new_path.exists() or paired_path.exists():
        raise FileExistsError("GG或配对产物已存在，不重新选标或覆盖；请用新目录重跑")
    old_payload = import_old_selection(old_path, old_copy_path)
    from features import load_cache
    cache = load_cache("paderborn", _project(config["cache_dir"]))
    keys, matrix = unit_level_features(cache)
    payload = select_gg(keys, matrix, gg_cfg)
    payload["cache_version"] = cache.version
    payload["repr_config"] = cache.repr_config
    payload["diagnostics"] = diagnostics(cache, payload["budgets"])
    _write_json(new_path, payload)
    paired = paired_splits(old_payload, payload)
    _write_json(paired_path, paired)
    return {"mode": "execute", "k_selected": payload["k_selected"],
            "cluster_sizes": payload["cluster_sizes"],
            "invalid_k": payload["invalid_k"],
            "T_pair": len(paired["T_pair"]),
            "common_unseen_bearings": paired["common_unseen_bearings"],
            "gg_output": str(new_path), "paired_output": str(paired_path),
            "old_copy": str(old_copy_path)}


def main():
    parser = argparse.ArgumentParser(
        description="v8 选标：旧池只读复制、GG 新池与配对留出（无训练、不加载模型）")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage3.json"))
    parser.add_argument("--execute", action="store_true",
                        help="实际执行旧池复制与 GG 计算；缺省只做 dry-run 检查")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run_selection(config, execute=args.execute)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if summary["mode"] == "execute":
        print(f"GG 完成：k_selected={summary['k_selected']}，T_pair={summary['T_pair']} 单元；"
              f"无效 k={summary['invalid_k'] or '无'}")


if __name__ == "__main__":
    main()
