# -*- coding: utf-8 -*-
"""探索性 q_phys：冻结 c=0.25，复用 v2 的 R，不替换正式 v1。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import evaluate
from features_v2 import CACHE_VERSION_V2

ROOT = Path(__file__).resolve().parent
RATIO_RULE = {
    "version": "q_phys_ratio_c025_v1",
    "c": 0.25,
    "class_order": ["Normal", "Ball", "IR", "OR"],
    "q_formula": "r=e_IR/(e_IR+e_OR); q=[0.25,0.25,0.5*r,0.5*(1-r)]",
    "zero_energy": "e_IR=e_OR=0 时 r=0.5，输出四类均匀分布",
    "aggregation": "未映射分析窗→记录→单元等权平均；最后 argmax，同分取类序最前者",
    "input_cache_version": CACHE_VERSION_V2,
    "input_note": "既有 v2 R：分频段噪底与 BPFI 边带扩展；不重算特征",
    "calibration": "c 固定，不拟合 alpha，不使用健康基线，不继续扫描 c",
    "limitation": "不独立识别 Normal/Ball，不保证正常样本中立或融合增益",
}


def q_phys_ratio(R):
    """从非空 (A,16) 非负实数 R 构造 c=0.25 的四类软证据。"""
    if np.iscomplexobj(R):
        raise ValueError("原始 R 必须是实数")
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 16 or not len(R):
        raise ValueError("q_phys_ratio 要求非空 (A,16) 原始 R")
    if not np.isfinite(R).all() or (R < 0).any():
        raise ValueError("原始 R 必须有限且非负")
    # 先共同缩放八项，再求两组份额，兼顾极大值与极小非零能量。
    scale = R[:, :8].max(axis=1)
    normalized = np.divide(R[:, :8], scale[:, None], out=np.zeros((len(R), 8)),
                           where=scale[:, None] > 0)
    outer, ir = normalized[:, :4].sum(axis=1), normalized[:, 4:8].sum(axis=1)
    total = ir + outer
    share_ir = np.divide(ir, total, out=np.full_like(ir, 0.5), where=total > 0)
    share_or = np.divide(outer, total, out=np.full_like(outer, 0.5), where=total > 0)
    return np.column_stack([np.full(len(R), 0.25), np.full(len(R), 0.25),
                            0.5 * share_ir, 0.5 * share_or])


def summarize_evidence(rows):
    """四类描述统计与逐样本 Normal 推力；无真值记录只计数量。"""
    result = {"total_count": len(rows), "unlabeled_count": sum(r["label"] is None for r in rows),
              "classes": {}}
    for label, name in enumerate(RATIO_RULE["class_order"]):
        probs = np.asarray([r["probabilities"] for r in rows if r["label"] == label], dtype=np.float64)
        count = len(probs)
        result["classes"][name] = {
            "support": count,
            "hits": int((probs.argmax(axis=1) == label).sum()) if count else 0,
            "recall": float((probs.argmax(axis=1) == label).mean()) if count else None,
            "q_mean": probs.mean(axis=0).tolist() if count else None,
        }
        if label == 0 and count:
            push = probs[:, 1:].max(axis=1) - probs[:, 0]
            result["normal_push"] = {
                "mean": float(push.mean()), "median": float(np.median(push)),
                "p95": float(np.quantile(push, 0.95, method="linear")), "max": float(push.max()),
                "push_of_class_mean": float(probs.mean(axis=0)[1:].max() - probs.mean(axis=0)[0]),
            }
    return result


def run_phys_ies_v2(config, execute=False, dataset="paderborn"):
    """独立导出探索软证据；不修改旧硬锚，不进行任何模型拟合。"""
    if dataset not in ("cwru", "paderborn"):
        raise ValueError(f"不支持的数据集：{dataset}")
    cache_dir = evaluate._project(config.get("cache_dir_v2", "outputs/features_v2"))
    output = evaluate._project(config.get(
        "phys_ies_v2_output", f"outputs/analysis_ies_diag/q_phys_ratio_c025/{dataset}"))
    if not output.resolve().is_relative_to((ROOT / "outputs" / "analysis_ies_diag").resolve()):
        raise ValueError("探索产物只能写入 v8 outputs/analysis_ies_diag/，禁止写入旧版或正式目录")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"探索输出目录非空，请另取 _retry1 路径：{output}")
    if not execute:
        return {"mode": "dry_run", "dataset": dataset, "output": str(output),
                "cache_ready": (cache_dir / f"{dataset}_ies_analysis.npy").is_file(),
                "rule": RATIO_RULE}
    meta = json.loads((cache_dir / f"{dataset}.json").read_text(encoding="utf-8"))
    if meta.get("version") != CACHE_VERSION_V2 or meta.get("dataset") != dataset:
        raise ValueError("缓存版本或数据集与冻结 v2 输入不一致")
    analysis = np.load(cache_dir / f"{dataset}_ies_analysis.npy", mmap_mode="r")
    if analysis.ndim != 2 or analysis.shape[1] != 16 or not len(analysis):
        raise ValueError("缓存 R 必须为非空 (A,16)")
    records, offset = [], 0
    for rec in meta["records"]:
        start, stop = rec["analysis_start"], rec["analysis_stop"]
        if (type(start) is not int or type(stop) is not int or start != offset
                or stop <= start or stop > len(analysis) or rec["dataset"] != dataset):
            raise ValueError(f"记录分析窗区间或数据集错误：{rec['record_key']}")
        q = q_phys_ratio(analysis[start:stop]).mean(axis=0)
        records.append({**rec, "probabilities": q.tolist(), "window_count": stop - start})
        offset = stop
    if offset != len(analysis):
        raise ValueError("记录区间未覆盖全部 R 分析窗")
    units = evaluate.aggregate_units(records)
    info = {"run_kind": "exploratory", "target_labels_seen": True,
            "formula_version": RATIO_RULE["version"], "cache_version": meta["version"],
            "cache_dir": str(cache_dir), "dataset": dataset,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_scope": "全量描述统计，不是分折源验证结果",
            "note": "已查看 PU 标签后选择的探索变体，不是盲测，不替换 v1"}
    diagnostics = {**info, "record_level": summarize_evidence(records),
                   "unit_level": summarize_evidence(units)}
    q_records = [{"record_key": r["record_key"], "unit_key": r["unit_key"], "label": r["label"],
                  "analysis_window_count": r["window_count"], "q_phys": r["probabilities"]} for r in records]
    q_units = [{"unit_key": r["unit_key"], "label": r["label"], "record_count": r["record_count"],
                "analysis_window_count": r["window_count"], "q_phys": r["probabilities"]} for r in units]
    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
            ("rule.json", {**info, **RATIO_RULE}),
            ("q_phys.json", {**info, "rule": RATIO_RULE, "records": q_records, "units": q_units}),
            ("diagnostics.json", diagnostics)):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return {"mode": "execute", "output": str(output), "record_count": len(records),
            "unit_count": len(units), "diagnostics": diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="探索性 q_phys：冻结 c=0.25，复用 v2 R")
    parser.add_argument("--dataset", choices=("cwru", "paderborn"), default="paderborn", help="数据集")
    parser.add_argument("--config", help="可选配置，只读取探索缓存与输出路径")
    parser.add_argument("--output", help="独立输出目录；非空不覆盖")
    parser.add_argument("--execute", action="store_true", help="重算软证据，不训练")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if args.output:
        config["phys_ies_v2_output"] = args.output
    print(json.dumps(run_phys_ies_v2(config, execute=args.execute, dataset=args.dataset),
                     ensure_ascii=False, indent=2))
