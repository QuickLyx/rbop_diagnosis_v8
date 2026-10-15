# -*- coding: utf-8 -*-
"""CWRU v2 缓存生成 + 源域 q_phys 方向评估。

目的：融合权重在源验证上选择，需要确认 v2 的 q_phys 在 CWRU 上是否
      对真 Normal 给出正确方向（这是"物理补 Normal"设计意图的前提）。

只写 outputs/features_v2/，不触碰任何 v1 正式产物。
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets"))

from features import (  # noqa: E402
    _enumerate_recordings, _record_signal_12k, _geometry_for, mapping_detail,
    extract_logspec, build_repr_config, EXPECTED_COUNTS, PHYS_CONFIG,
)
from features_v2 import (  # noqa: E402
    ies_features_v2, physical_features_v2, CACHE_VERSION_V2, IES_RULE_V2,
)
from bearing_facts import LABEL_NAMES  # noqa: E402

DATA_ROOT = ROOT.parent / "rbop_diagnosis_v6" / "Data"
OUT = ROOT / "outputs" / "features_v2"
DATASET = "cwru"


def build_cwru_v2():
    """生成 CWRU v2 缓存（v2 IES + 25 维摘要 + logspec 保持冻结口径）。"""
    recordings = list(_enumerate_recordings(DATASET, DATA_ROOT))
    expected = EXPECTED_COUNTS[DATASET]
    print(f"CWRU 记录数: {len(recordings)}（预期 {expected['records']}）")

    X_list, phys_list, ies_list, records = [], [], [], []
    logspec_offset, analysis_offset = 0, 0

    for i, rec in enumerate(recordings, 1):
        signal = _record_signal_12k(rec)
        n = signal.size // 512
        windows = signal[: n * 512].reshape(n, 512)
        logspec = extract_logspec(windows, build_repr_config())

        orders, rotation_hz, rpm, rpm_source = _geometry_for(rec)
        ies = ies_features_v2(signal, orders, rotation_hz)
        phys = physical_features_v2(signal, ies)

        if len(ies) != max(0, (512 * n - 2048) // 1024 + 1):
            raise ValueError(f"分析窗数不符：{rec.record_key}")
        if len(phys) != n:
            raise ValueError(f"phys25 长度不符：{rec.record_key}")
        _, map_ties, map_outside = mapping_detail(n, len(ies))

        X_list.append(logspec)
        phys_list.append(phys)
        ies_list.append(ies)
        records.append({
            **asdict(rec), "path": str(rec.path),
            "start": logspec_offset, "stop": logspec_offset + n,
            "analysis_start": analysis_offset, "analysis_stop": analysis_offset + len(ies),
            "rpm": rpm, "rpm_source": rpm_source,
            "map_ties": map_ties, "map_outside": map_outside,
            "constant_windows": int((windows.var(axis=1) == 0).sum()),
        })
        logspec_offset += n
        analysis_offset += len(ies)
        if i % 50 == 0:
            print(f"  {i}/{len(recordings)}")

    X = np.concatenate(X_list).astype(np.float32)[:, None, :]
    phys = np.concatenate(phys_list).astype(np.float32)
    ies = np.concatenate(ies_list).astype(np.float32)

    assert len(X) == expected["windows"], f"窗数 {len(X)} != {expected['windows']}"
    assert len({r["unit_key"] for r in records}) == expected["units"]
    for arr, name in ((X, "logspec"), (phys, "phys25"), (ies, "ies_analysis")):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 含非有限值")

    OUT.mkdir(parents=True, exist_ok=True)
    for path, arr in ((OUT / f"{DATASET}.npy", X),
                      (OUT / f"{DATASET}_phys25.npy", phys),
                      (OUT / f"{DATASET}_ies_analysis.npy", ies)):
        if path.exists():
            raise FileExistsError(f"已存在，拒绝覆盖：{path}")
        np.save(path, arr)

    payload = {
        "dataset": DATASET, "version": CACHE_VERSION_V2, "script": "features_v2.py",
        "repr_config": build_repr_config(), "phys_config": PHYS_CONFIG,
        "revision_v2": IES_RULE_V2["revision_v2"],
        "counts": {"records": len(records), "units": expected["units"], "windows": len(X)},
        "records": records,
    }
    (OUT / f"{DATASET}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 CWRU v2：{len(X)} 窗 / {len(ies)} 分析窗 / {expected['units']} 单元")
    return payload


def load_v2(dataset):
    folder = OUT
    X = np.load(folder / f"{dataset}.npy", mmap_mode="r")
    phys = np.load(folder / f"{dataset}_phys25.npy", mmap_mode="r")
    ies = np.load(folder / f"{dataset}_ies_analysis.npy", mmap_mode="r")
    meta = json.loads((folder / f"{dataset}.json").read_text(encoding="utf-8"))
    return SimpleNamespace(X=X, phys25=phys, ies_analysis=ies,
                           records=meta["records"], version=meta["version"])


def fit_alpha(records, ies, label=0):
    """alpha = sqrt(e_IR·e_OR) 在指定类记录上的均值（逐窗）。"""
    vals = []
    per_record = []
    for rec in records:
        if rec["label"] != label:
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        if not len(R):
            continue
        e_or = R[:, 0:4].mean(axis=1)
        e_ir = R[:, 4:8].mean(axis=1)
        geom = np.sqrt(e_ir * e_or)
        vals.append(geom)
        per_record.append((rec["unit_key"], float(geom.mean())))
    if not vals:
        raise ValueError("无可用记录")
    return float(np.concatenate(vals).mean()), per_record


def q_phys_of(R, alpha):
    """q_phys 软分（v2 公式）。"""
    e_or = R[:, 0:4].mean(axis=1)
    e_ir = R[:, 4:8].mean(axis=1)
    v = np.column_stack([np.full(len(R), alpha), np.full(len(R), alpha),
                         (alpha + e_ir) / (1.5 * (alpha + e_or)),
                         (alpha + e_or) / (1.5 * (alpha + e_ir))])
    return v / v.sum(axis=1, keepdims=True)


def hard_metrics(records, ies, label_set=(0, 2, 3)):
    """记录级硬判召回（v2 R）。"""
    stats = {l: [0, 0] for l in label_set}
    for rec in records:
        lab = rec["label"]
        if lab not in stats:
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        e_or = R[:, 0:4].mean(axis=1)
        e_ir = R[:, 4:8].mean(axis=1)
        hard = np.zeros(len(R), dtype=int)
        hard[e_or > 1.5 * e_ir] = 3
        hard[e_ir > 1.5 * e_or] = 2
        pred = int(np.bincount(hard, minlength=4).argmax())
        stats[lab][1] += 1
        stats[lab][0] += int(pred == lab)
    return stats


def q_by_label(records, ies, alpha):
    """每类 q_phys 平均 + IR 分量排名情况。"""
    store = {}
    for rec in records:
        lab = rec["label"]
        if lab not in (0, 1, 2, 3):
            continue
        R = np.asarray(ies[rec["analysis_start"]:rec["analysis_stop"]], dtype=np.float64)
        q = q_phys_of(R, alpha).mean(axis=0)
        store.setdefault(lab, []).append(q)
    return {l: np.mean(v, axis=0) for l, v in store.items()}


def main():
    print("=" * 60)
    print("阶段 1：生成 CWRU v2 缓存")
    print("=" * 60)
    if not (OUT / f"{DATASET}.json").is_file():
        build_cwru_v2()
    else:
        print("CWRU v2 缓存已存在，跳过生成")

    print("\n" + "=" * 60)
    print("阶段 2：v2 R 上重新拟合 alpha（源域 Normal）")
    print("=" * 60)
    cache = load_v2(DATASET)
    print(f"缓存版本 {cache.version}，记录 {len(cache.records)}")
    alpha_v2, per_record = fit_alpha(cache.records, cache.ies_analysis, label=0)
    print(f"alpha_v2 = {alpha_v2:.4f}（v1 R 上是 2.3170）")
    for unit, val in sorted(per_record):
        print(f"  {unit}: {val:.4f}")

    print("\n" + "=" * 60)
    print("阶段 3：CWRU v2 硬判召回")
    print("=" * 60)
    stats = hard_metrics(cache.records, cache.ies_analysis)
    for lab in (0, 2, 3):
        ok, tot = stats[lab]
        print(f"  {LABEL_NAMES[lab]:>11}: {ok:>3}/{tot:<3} = {100*ok/max(tot,1):>5.1f}%")

    print("\n" + "=" * 60)
    print("阶段 4：q_phys 四类方向（关键）")
    print("=" * 60)
    for name, alpha in (("alpha=1.0（v1 隐含）", 1.0), (f"alpha={alpha_v2:.3f}（v2 拟合）", alpha_v2)):
        qm = q_by_label(cache.records, cache.ies_analysis, alpha)
        print(f"\n  {name}:")
        for lab in (0, 1, 2, 3):
            if lab not in qm:
                continue
            m = qm[lab]
            top = int(np.argmax(m))
            mark = " <-正确" if top == lab else ""
            print(f"    {LABEL_NAMES[lab]:>11}: [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]"
                  f"  最高={LABEL_NAMES[top]}{mark}")

    result = {
        "alpha_v1_r": 2.316955, "alpha_v2_r": alpha_v2,
        "hard": {LABEL_NAMES[l]: [stats[l][0], stats[l][1]] for l in (0, 2, 3)},
        "q_alpha1": {LABEL_NAMES[l]: q_by_label(cache.records, cache.ies_analysis, 1.0)[l].tolist()
                     for l in (0, 1, 2, 3)
                     if l in q_by_label(cache.records, cache.ies_analysis, 1.0)},
        "q_alpha_v2": {LABEL_NAMES[l]: q_by_label(cache.records, cache.ies_analysis, alpha_v2)[l].tolist()
                       for l in (0, 1, 2, 3)
                       if l in q_by_label(cache.records, cache.ies_analysis, alpha_v2)},
    }
    out_path = ROOT / "outputs" / "analysis_ies_diag" / "cwru_v2_source.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存：{out_path}")


if __name__ == "__main__":
    main()
