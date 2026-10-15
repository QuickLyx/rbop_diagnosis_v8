# -*- coding: utf-8 -*-
"""v8 两个物理锚：旧包络锚只读重评分、新 IES 锚 phys_ies 与 q_phys（计划 §7）。

旧锚（phys_envelope）：只读复用 v7 的 predictions.csv/unit_predictions.csv，
按 v8 配对清单重新评分，不改规则、不重新生成预测。新锚（phys_ies）：零拟合
零标签规则，逐 2048/1024 分析窗 IES 硬判 OR/IR/Normal，并给出融合用 q_phys
软分；Ball 恒零是候选空间限制，不是拒识。两锚规则互不污染。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import evaluate
from features import ies_features  # noqa: F401  接口约定的 IES 信号核心入口（features 侧）

ROOT = Path(__file__).resolve().parent
PROTOCOL = "v8_stage3"
LEGACY_CACHE_VERSION = "v7_logspec512_12k_drop_ka08_v1"

#: 新 IES 锚冻结规则（计划 §7.2/§7.3）；configs 的 physical 块必须逐项一致。
IES_RULE = {
    "analysis_window": 2048, "analysis_stride": 1024,
    "bandpass_hz": [2000.0, 5000.0], "bandpass_order": 4,
    "harmonics": [1, 2, 3, 4], "order_names": ["BPFO", "BPFI", "BSF", "FTF"],
    "hard_rule": "e_OR > 1.5*e_IR 判 OR(3)；e_IR > 1.5*e_OR 判 IR(2)；否则 Normal(0)；Ball 恒零",
    "ies_ratio_threshold": 1.5, "q_pseudocount": 1,
    "q_formula": "v=[1,1,(1+e_IR)/(1.5*(1+e_OR)),(1+e_OR)/(1.5*(1+e_IR))]；q_phys=v/sum(v)",
    "aggregation": "分析窗→记录→单元等权平均；映射摘要不重复计数",
    "training": "零拟合零标签：不用目标数据改阈值/频带/谐波/窗长",
    "ball_note": "Ball 恒零属候选空间限制，不证明跨任务 Ball 拒识",
}


def ies_evidence(R):
    """逐分析窗 IES 证据：返回 (hard_onehot, q_phys, e_ir, e_or)。

    `R` 为 features.ies_features 的 (A,16) 原始 R（BPFO/BPFI/BSF/FTF 各 4 谐波）。
    硬判用 e_OR/e_IR 的均值；同阈值边界/两者全零均判 Normal（行 0/1/3 判 Normal）。
    q_phys 四项恒正，v=[1,1,(1+e_IR)/(1.5(1+e_OR)),(1+e_OR)/(1.5(1+e_IR))]。
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 16 or not len(R):
        raise ValueError("ies_evidence 要求非空 (A,16) 原始 R")
    if not np.isfinite(R).all() or (R < 0).any():
        raise ValueError("原始 R 必须有限且非负")
    e_or = R[:, 0:4].mean(axis=1)
    e_ir = R[:, 4:8].mean(axis=1)
    hard = np.zeros(len(R), dtype=np.int64)  # 默认 Normal；Ball 恒零
    hard[e_or > 1.5 * e_ir] = 3
    hard[e_ir > 1.5 * e_or] = 2
    onehot = np.zeros((len(R), 4), dtype=np.float64)
    onehot[np.arange(len(R)), hard] = 1.0
    v = np.column_stack([np.ones(len(R)), np.ones(len(R)),
                         (1 + e_ir) / (1.5 * (1 + e_or)),
                         (1 + e_or) / (1.5 * (1 + e_ir))])
    q_phys = v / v.sum(axis=1, keepdims=True)
    if not np.isfinite(q_phys).all() or (q_phys <= 0).any():
        raise ValueError("q_phys 必须有限且恒正")
    return onehot, q_phys, e_ir, e_or


def _record_ies_evidence(record, cache):
    """单记录逐分析窗证据；直接用本记录的 ies_analysis 区间，不重复映射。"""
    R = np.asarray(cache.ies_analysis[record["analysis_start"]:record["analysis_stop"]])
    if not len(R):
        raise ValueError(f"记录没有完整 2048 分析窗：{record['record_key']}")
    return ies_evidence(R)


def _expected_analysis_windows(record):
    """冻结映射给出的该记录分析窗数（512/512 logspec 窗数 → 2048/1024 完整窗数）。"""
    from features import analysis_window_count
    return analysis_window_count(record["stop"] - record["start"])


def run_phys_ies(config, execute=False):
    """新 IES 锚：硬判、q_phys 与按配对清单的重评分（计划 §7.2/§7.3）。

    dry-run 只检查配置与输入存在性；execute 全量逐记录计算并落盘，
    非空输出目录不覆盖。
    """
    physical = config.get("physical") or {}
    for key, value in (("analysis_window", 2048), ("analysis_stride", 1024),
                       ("ies_ratio_threshold", 1.5), ("q_pseudocount", 1)):
        if physical.get(key) != value:
            raise ValueError(f"configs 的 physical.{key}={physical.get(key)!r} 偏离冻结值 {value!r}")
    cache_dir = evaluate._project(config["cache_dir"])
    output = evaluate._project(config.get("phys_ies_output", "outputs/eval/phys_ies"))
    if not execute:
        return {"mode": "dry_run", "cache_dir": str(cache_dir), "output": str(output),
                "cache_ready": (cache_dir / "paderborn_ies_analysis.npy").is_file()}
    from features import load_cache
    cache = load_cache("paderborn", cache_dir)
    if cache.version != config["cache_version"]:
        raise ValueError(f"缓存口径 {cache.version} 与冻结配置不一致")
    selections = _paired_lists(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"phys_ies 输出目录非空，重跑请换目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    unit_rows = []
    for record in cache.records:
        n_windows = record["stop"] - record["start"]
        n_analysis = record["analysis_stop"] - record["analysis_start"]
        if n_analysis != _expected_analysis_windows(record):
            raise ValueError(f"记录分析窗数与冻结映射不一致：{record['record_key']}")
        onehot, q_phys, e_ir, e_or = _record_ies_evidence(record, cache)
        hard_prob = onehot.mean(axis=0)
        soft_prob = q_phys.mean(axis=0)
        records.append({**record,
                        "start": record["analysis_start"], "stop": record["analysis_stop"],
                        "probabilities": hard_prob.tolist(),
                        "q_phys": soft_prob.tolist(),
                        "e_ir": float(e_ir.mean()), "e_or": float(e_or.mean()),
                        "logspec_window_count": n_windows, "window_count": n_analysis})
    unit_rows = aggregate_by_unit(records, use_q_phys=False)
    with (output / "rule.json").open("x", encoding="utf-8") as handle:
        json.dump(IES_RULE, handle, ensure_ascii=False, indent=2)
    info = {"protocol": PROTOCOL, "arm": "phys_ies", "dataset": "paderborn",
            "cache_version": cache.version, "repr_config": cache.repr_config,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_kind": "formal", "trained": False,
            "evaluation_scope": "zero_shot_all_primary",
            "rule": IES_RULE,
            "evaluation_splits": {name: units for name, units in selections.items()},
            "note": "新锚为 N/IR/OR 三类规则，Ball 恒零属候选空间限制；"
                    "q_phys 是融合用软分，不是新锚硬判结果"}
    evaluate.save_reports(records, unit_rows, output, pool_units=[],
                          eval_info=info)
    # 硬判之外的融合软分与比值摘要单独落盘，不混入锚预测 CSV。
    q_records = [{"record_key": row["record_key"], "unit_key": row["unit_key"],
                  "q_phys": row["q_phys"], "e_ir": row["e_ir"], "e_or": row["e_or"],
                  "analysis_window_count": row["window_count"]} for row in records]
    q_units = [{"unit_key": row["unit_key"], "q_phys": _unit_q_phys(records, row["unit_key"])}
               for row in unit_rows]
    with (output / "q_phys.json").open("x", encoding="utf-8") as handle:
        json.dump({"records": q_records, "units": q_units, "rule": IES_RULE},
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
    return {"mode": "execute", "output": str(output),
            "record_count": len(records), "unit_count": len(unit_rows)}


def _unit_q_phys(records, unit_key):
    """单元级 q_phys：记录级软分等权平均（与硬判同一条记录→单元管道）。"""
    rows = [np.asarray(row["q_phys"], dtype=np.float64) for row in records
            if row["unit_key"] == unit_key]
    return np.mean(rows, axis=0).tolist()


def aggregate_by_unit(rows, use_q_phys=False):
    """物理行（已是记录级概率）按记录→单元等权平均；身份校验同 evaluate.aggregate_units。"""
    key_field = "q_phys" if use_q_phys else "probabilities"
    normalized = [{**row, "probabilities": row[key_field]} for row in rows]
    units = evaluate.aggregate_units(normalized)
    for unit in units:
        unit["logspec_window_count"] = sum(row["logspec_window_count"] for row in rows
                                           if row["unit_key"] == unit["unit_key"])
    return units


def _paired_lists(config):
    """读取并校验配对清单，返回 PU 评估 splits 的单元列表（与 evaluate 同源）。"""
    paired = evaluate._read_json(evaluate._project(config["paired_splits_path"]))
    from selection import load_selection, paired_splits
    old = load_selection(evaluate._project(config["selection_path"]))
    gg = load_selection(evaluate._project(config["selection_gg_path"]))
    expected = paired_splits(old, gg)
    if any(paired.get(key) != value for key, value in expected.items()):
        raise ValueError("配对清单与冻结新旧池不一致")
    return {"all_primary": expected["U"], "T_old": expected["T_old"], "T_gg": expected["T_gg"],
            "T_pair": expected["T_pair"], "common_unseen_units": expected["common_unseen_units"]}


def rescore_old_anchor(config, execute=False):
    """旧锚只读重评分：读 v7 predictions/unit_predictions CSV，按 v8 清单重评。

    不重新计算预测、不修改 v7 目录；缺旧预测或单元覆盖不全时报错。
    """
    old_dir = evaluate._project(config["old_physical_path"])
    output = evaluate._project(config.get("phys_envelope_output", "outputs/eval/phys_envelope"))
    if not execute:
        return {"mode": "dry_run", "old_dir": str(old_dir), "output": str(output),
                "old_predictions": (old_dir / "predictions.csv").is_file(),
                "old_unit_predictions": (old_dir / "unit_predictions.csv").is_file()}
    for name in ("predictions.csv", "unit_predictions.csv", "rule.json"):
        if not (old_dir / name).is_file():
            raise FileNotFoundError(f"旧锚预测缺失，不静默重新生成：{old_dir / name}")
    record_rows = evaluate.read_predictions(old_dir / "predictions.csv")
    unit_rows = evaluate.read_predictions(old_dir / "unit_predictions.csv", unit=True)
    _check_old_anchor_rows(record_rows, unit_rows, old_dir)
    selections = _paired_lists(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"旧锚重评分输出目录非空，重跑请换目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(old_dir / "rule.json", output / "rule.json")
    info = {"protocol": PROTOCOL, "arm": "phys_envelope", "dataset": "paderborn",
            "cache_version": config["cache_version"],
            "old_cache_version": LEGACY_CACHE_VERSION,
            "reused_from": str(old_dir),
            "reused_artifacts": ["predictions.csv", "unit_predictions.csv"],
            "rule_source": "v7 outputs/eval/phys_envelope/rule.json（原规则未改动）",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_kind": "formal", "trained": False,
            "evaluation_scope": "zero_shot_all_primary",
            "evaluation_splits": {name: units for name, units in selections.items()},
            "note": "旧完整记录锚只读复用 v7 预测，本目录只重评分；不是新算法"}
    evaluate.save_reports(record_rows, unit_rows, output, pool_units=[],
                          eval_info=info)
    return {"mode": "execute", "output": str(output),
            "record_count": len(record_rows), "unit_count": len(unit_rows)}


def _check_old_anchor_rows(record_rows, unit_rows, old_dir):
    """旧锚复用前的完整性检查：记录数、单元覆盖与概率有效性。"""
    if len(record_rows) != 2559 or len(unit_rows) != 128:
        raise ValueError(f"旧锚预测计数不符：记录 {len(record_rows)}（预期 2559）、"
                         f"单元 {len(unit_rows)}（预期 128）：{old_dir}")
    units = {row["unit_key"] for row in unit_rows}
    missing = [row["unit_key"] for row in record_rows if row["unit_key"] not in units]
    if missing:
        raise ValueError(f"旧锚单元预测未覆盖全部记录单元：{missing[:3]}")
    primary = {row["unit_key"] for row in record_rows if row["label"] is not None}
    if len(primary) != 116:
        raise ValueError(f"旧锚主口径单元数不符：{len(primary)} != 116")
    recomputed = {row["unit_key"]: row for row in evaluate.aggregate_units(record_rows)}
    if len(units) != 128 or set(recomputed) != units:
        raise ValueError("旧锚单元重复或记录覆盖与单元CSV不一致")
    from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
    expected = {f"{b}/{c}" for b in PRIMARY_BEARINGS for c in CONDITIONS}
    if primary != expected:
        raise ValueError("旧锚未恰好覆盖冻结116主单元")
    for row in unit_rows:
        actual = recomputed[row["unit_key"]]
        if actual["label"] != row["label"] or not np.allclose(actual["probabilities"], row["probabilities"], atol=1e-6):
            raise ValueError("旧锚记录/单元真值或聚合概率不一致")
    from bearing_facts import BY_ID
    for row in record_rows + unit_rows:
        bearing = BY_ID.get(row.get("bearing_id"))
        if (row.get("dataset") != "paderborn" or bearing is None
                or row.get("condition") not in CONDITIONS or row.get("label") != bearing.label
                or row.get("unit_key") != f"{bearing.bearing_id}/{row['condition']}"):
            raise ValueError(f"旧锚记录身份或真值与冻结事实不符：{row.get('record_key', row.get('unit_key'))}")


def run(config, arm, execute=False):
    """按臂分发：phys_envelope 旧锚重评分，phys_ies 新锚。"""
    target_key = "phys_envelope_output" if arm == "phys_envelope" else "phys_ies_output"
    target = evaluate._project(config.get(target_key, f"outputs/eval/{arm}"))
    if execute and not target.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("物理锚产物只允许写入v8/outputs")
    if arm == "phys_envelope":
        return rescore_old_anchor(config, execute)
    if arm == "phys_ies":
        return run_phys_ies(config, execute)
    raise ValueError(f"未知物理锚臂：{arm}（只支持 phys_envelope / phys_ies）")


def _self_check():
    """规则自检：阈值边界、全零判 Normal、q_phys 恒正且和为 1。"""
    R = np.zeros((3, 16))
    R[0, 0:4] = 3.0  # e_OR=3
    R[0, 4:8] = 2.0  # e_IR=2 → 3 > 1.5*2 为假（边界）→ Normal
    R[1, 0:4] = 3.0001
    R[1, 4:8] = 2.0  # 严格大于 → OR
    R[2, 0:4] = 2.0
    R[2, 4:8] = 3.0  # e_IR=3 > 1.5*2=3 为假（边界）→ Normal；严格大于才判 IR
    onehot, q, e_ir, e_or = ies_evidence(R)
    assert onehot.argmax(axis=1).tolist() == [0, 3, 0]
    assert onehot[0, 0] == 1.0 and onehot[:, 1].sum() == 0
    assert np.allclose(q.sum(axis=1), 1) and (q > 0).all()
    zero, qz, _, _ = ies_evidence(np.zeros((1, 16)))
    assert zero.argmax() == 0 and abs(qz.sum() - 1) < 1e-12
    with np.errstate(invalid="ignore"):
        try:
            ies_evidence(np.full((1, 16), np.nan))
        except ValueError:
            pass
        else:
            raise AssertionError("非有限 R 必须报错")
    print("IES 规则自检通过（未读取真实数据）")


def main():
    parser = argparse.ArgumentParser(description="v8 物理锚：旧锚只读重评分 / 新 IES 锚")
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage3.json"))
    parser.add_argument("--arm", choices=("phys_envelope", "phys_ies"), required=True)
    parser.add_argument("--execute", action="store_true",
                        help="实际执行计算并落盘；缺省只做 dry-run 检查")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run(config, args.arm, execute=args.execute)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
