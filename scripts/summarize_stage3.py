# -*- coding: utf-8 -*-
"""只读现有 JSON/CSV 汇总固定矩阵，不以目录存在冒充完成。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oneclass import dump_json, load_config, new_output, project_path


def statistics(values):
    array = np.asarray(values, dtype=float)
    return {"count": len(array), "mean": float(array.mean()) if len(array) else None,
            "std_ddof1": float(array.std(ddof=1)) if len(array) > 1 else None,
            "min": float(array.min()) if len(array) else None,
            "max": float(array.max()) if len(array) else None}


def validate_report(folder, info, metrics):
    """核对实际CSV覆盖与重算指标，不能仅凭四个文件存在报完成。"""
    from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
    from evaluate import aggregate_units, read_predictions, score_predictions
    expected = {f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS for condition in CONDITIONS}
    records = read_predictions(folder / "predictions.csv")
    units = read_predictions(folder / "unit_predictions.csv", unit=True)
    primary = [r for r in units if r["label"] is not None]
    if (len(primary) != 116 or {r["unit_key"] for r in primary} != expected
            or len(units) != len({r["unit_key"] for r in units})
            or set(info.get("evaluation_splits", {}).get("all_primary", [])) != expected):
        raise ValueError(f"报告未覆盖冻结116单元，不能计为完成：{folder}")
    recomputed = {r["unit_key"]: r for r in aggregate_units(records)}
    if set(recomputed) != {r["unit_key"] for r in units}:
        raise ValueError(f"记录和单元报告覆盖不一致：{folder}")
    for row in units:
        source = recomputed[row["unit_key"]]
        if source["label"] != row["label"] or not np.allclose(source["probabilities"], row["probabilities"], atol=1e-6):
            raise ValueError(f"记录和单元概率/真值不一致：{folder}")
    actual = score_predictions(primary)
    e1, e2 = actual["E1"]["macro_recall"], actual["E2"]["macro_recall"]
    d3 = metrics.get("D3", {})
    if (d3.get("n") != 116 or d3.get("passed") is not bool(e1 >= 0.8 and e2 >= 0.9)
            or d3.get("E1") is None or d3.get("E2") is None
            or not np.isclose(d3["E1"], e1) or not np.isclose(d3["E2"], e2)):
        raise ValueError(f"D3数量/指标与预测CSV不符：{folder}")


def discover_reports(eval_root, config):
    reports = {}
    for path in sorted(eval_root.glob("*/eval_info.json")):
        folder = path.parent
        if not all((folder / n).is_file() for n in ("metrics.json", "predictions.csv", "unit_predictions.csv")):
            continue
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("protocol") != "v8_stage3" or info.get("dataset") != "paderborn":
            continue
        arm, fold, seed = info.get("arm"), info.get("fold"), info.get("seed")
        auxiliary = [f"{base}_{ablation}" for base in ("fusion_joint", "fusion_ce_K19")
                     for ablation in config["fusion"]["ablations"]]
        if arm in config["physical_arms"]:
            fold, seed = None, None
        elif arm not in config["arms"] + auxiliary or fold not in config["folds"] or seed not in config["seeds"]:
            continue
        if info.get("run_kind") == "pilot" or "pilot" in folder.parts:
            raise ValueError(f"pilot不能冒充正式评估结果：{folder}")
        if info.get("cache_version") != config["cache_version"]:
            raise ValueError(f"报告缓存口径不匹配：{folder}")
        key = (arm, fold, seed)
        if key in reports:
            raise ValueError(f"同格存在多个报告，请显式选定正式结果，不能按得分择优：{key}")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        validate_report(folder, info, metrics)
        reports[key] = {"folder": str(folder), "info": info, "metrics": metrics}
    return reports


def collect_errors(folder):
    from evaluate import read_predictions
    rows = read_predictions(Path(folder) / "unit_predictions.csv", unit=True)
    primary = [r for r in rows if r["dataset"] == "paderborn" and r["label"] is not None]
    errors = [r for r in primary if int(np.argmax(r["probabilities"])) != r["label"]]
    result = {"unit_count": len(primary), "error_count": len(errors),
              "P_ball": float(np.mean([np.argmax(r["probabilities"]) == 1 for r in primary])) if primary else None,
              "error_unit_keys": [r["unit_key"] for r in errors]}
    for field in ("label", "condition", "origin", "bearing_id"):
        result[f"errors_by_{field}"] = dict(Counter(str(r.get(field)) for r in errors))
    result["prediction_counts"] = dict(Counter(str(int(np.argmax(r["probabilities"]))) for r in primary))
    return result


def summarize(config, reports):
    rows, arms, missing, errors = [], {}, [], {}
    for arm in config["arms"]:
        arm_rows = []
        for fold in config["folds"]:
            for seed in config["seeds"]:
                key = (arm, fold, seed)
                if key not in reports:
                    missing.append({"arm": arm, "fold": fold, "seed": seed})
                    continue
                item = reports[key]
                d3 = item["metrics"]["D3"]
                row = {"arm": arm, "fold": fold, "seed": seed, "E1": d3["E1"],
                       "E2": d3["E2"], "passed": d3["passed"], "n": d3["n"], "folder": item["folder"]}
                rows.append(row)
                arm_rows.append(row)
                if d3["passed"] is False:
                    errors[f"{arm}_f{fold}_s{seed}"] = collect_errors(item["folder"])
        arms[arm] = {"completed": len(arm_rows), "planned": 12,
                     "E1": statistics([r["E1"] for r in arm_rows if r["E1"] is not None]),
                     "E2": statistics([r["E2"] for r in arm_rows if r["E2"] is not None]),
                     "pass_fraction_of_completed": float(np.mean([r["passed"] for r in arm_rows if r["passed"] is not None]))
                         if any(r["passed"] is not None for r in arm_rows) else None,
                     "worst_E1_cell": min((r for r in arm_rows if r["E1"] is not None), key=lambda r: r["E1"], default=None)}
    paired = []
    comparisons = config["paired_chain"] + [[base, f"{base}_{ablation}"]
        for base in ("fusion_joint", "fusion_ce_K19") for ablation in config["fusion"]["ablations"]]
    for left, right, scope in [(a, b, scope) for a, b in comparisons
                              for scope in (("T_pair", "common_unseen_units")
                                  if {a, b} == {"ce_K19_gg", "ce_K19"} else ("T_old",))]:
        differences = []
        for fold in config["folds"]:
            for seed in config["seeds"]:
                a = reports.get((left, fold, seed)) or reports.get((left, None, None))
                b = reports.get((right, fold, seed)) or reports.get((right, None, None))
                if not a or not b:
                    continue
                sa = a["metrics"].get("splits", {}).get(scope)
                sb = b["metrics"].get("splits", {}).get(scope)
                if sa is None or sb is None:
                    raise ValueError(f"配对报告缺冻结共同清单：{left}/{right}/{scope}")
                la = a["info"]["evaluation_splits"][scope]
                lb = b["info"]["evaluation_splits"][scope]
                if set(la) != set(lb) or set(la) & set(a["info"].get("supervised_pool_units") or []) or set(lb) & set(b["info"].get("supervised_pool_units") or []):
                    raise ValueError("配对清单不同或包含监督池")
                ma, mb = sa["unit_level"], sb["unit_level"]
                if ma["n"] != mb["n"]:
                    raise ValueError("配对指标分母不同")
                differences.append({"fold": fold, "seed": seed, "n": ma["n"],
                    "delta_E1": None if not ma["n"] else ma["E1"]["macro_recall"]-mb["E1"]["macro_recall"],
                    "delta_E2": None if not ma["n"] else ma["E2"]["macro_recall"]-mb["E2"]["macro_recall"]})
        paired.append({"left": left, "right": right, "scope": scope, "cells": differences,
                       "delta_E1": statistics([d["delta_E1"] for d in differences if d["delta_E1"] is not None]),
                       "delta_E2": statistics([d["delta_E2"] for d in differences if d["delta_E2"] is not None])})
    auxiliary_keys = [(f"{base}_{ablation}", fold, seed)
                      for base in ("fusion_joint", "fusion_ce_K19") for ablation in config["fusion"]["ablations"]
                      for fold in config["folds"] for seed in config["seeds"]]
    return {"protocol": "v8_stage3", "counts": config["counts"], "main_completed": len(rows),
            "fusion_ablation_completed": sum(key in reports for key in auxiliary_keys),
            "fusion_ablation_cells": [{"arm": key[0], "fold": key[1], "seed": key[2],
                "D3": reports[key]["metrics"]["D3"], "folder": reports[key]["folder"]}
                for key in auxiliary_keys if key in reports],
            "fusion_ablation_missing": [list(key) for key in auxiliary_keys if key not in reports],
            "physical_anchor_completed": sum((arm, None, None) in reports for arm in config["physical_arms"]),
            "physical_anchors": {arm: reports[(arm, None, None)]["metrics"] for arm in config["physical_arms"] if (arm, None, None) in reports},
            "main_missing": missing, "cells": rows, "arms": arms, "paired": paired,
            "zero_compute_error_structure": errors,
            "interpretation": "全U指标含监督池，仅描述；泛化只看冻结共同留出。12格不是12个独立目标测试集，不作独立t检验。"}


def training_status(config, reports):
    """训练完成数与评估完成数分开，固定文件及元数据不符时不能报完成。"""
    result = []
    for arm in config["deep_arms"] + ["svm_phys"]:
        for fold in config["folds"]:
            for seed in config["seeds"]:
                report = reports.get((arm, fold, seed))
                reference = report["info"].get("checkpoint") if report else None
                folder = Path(reference).parent if reference else ROOT / f"outputs/train/{arm}_f{fold}_s{seed}"
                if arm == "svm_phys" and report:
                    folder = project_path(report["info"].get("model_dir", str(folder)))
                names = ["config.json", "split.json"] + (["model.joblib", "scaler.json", "fit_info.json", "calibration.json", "quotas.json"]
                    if arm == "svm_phys" else ["last.pt", "best.pt", "history.csv"])
                missing = [name for name in names if not (folder/name).is_file() or (folder/name).stat().st_size == 0]
                if not missing:
                    meta = json.loads((folder/"config.json").read_text(encoding="utf-8"))
                    split = json.loads((folder/"split.json").read_text(encoding="utf-8"))
                    if (meta.get("protocol") != config["protocol"] or meta.get("arm") != arm
                            or meta.get("fold") != fold or meta.get("seed") != seed
                            or meta.get("cache_version") != config["cache_version"] or meta.get("run_kind") != "formal"
                            or set(split["source"]["train"]) & set(split["source"]["val"])):
                        raise ValueError(f"训练产物版本/折/种子/split不符：{folder}")
                    if arm != "svm_phys":
                        import csv
                        with (folder/"history.csv").open(encoding="utf-8", newline="") as handle:
                            history = list(csv.DictReader(handle))
                        if len(history) != 30 or [int(row["epoch"]) for row in history] != list(range(1, 31)):
                            raise ValueError(f"正式训练history未完整30轮：{folder}")
                result.append({"arm": arm, "fold": fold, "seed": seed, "folder": str(folder),
                               "completed": not missing, "missing_files": missing})
    return {"cells": result,
            "gradient_completed": sum(r["completed"] for r in result if r["arm"] not in ("adabn", "svm_phys")),
            "adabn_completed": sum(r["completed"] for r in result if r["arm"] == "adabn"),
            "shallow_completed": sum(r["completed"] for r in result if r["arm"] == "svm_phys")}


def auxiliary_status(config):
    result = {}
    for fold in config["folds"]:
        folder = ROOT / f"outputs/train/normal_f{fold}"
        required = ["model.joblib", "scaler.json", "config.json", "split.json", "calibration.json"]
        missing = [name for name in required if not (folder / name).is_file()]
        if not missing:
            info = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
            if (info.get("protocol") != config["protocol"] or info.get("arm") != "normal"
                    or info.get("fold") != fold or info.get("cache_version") != config["cache_version"]
                    or info.get("run_kind") != "formal"
                    or set(split["train"]) & set(split["val"]) or len(set(split["train"]) | set(split["val"])) != 153):
                raise ValueError(f"Normal产物版本/划分不符：{folder}")
        result[f"normal_f{fold}"] = {"completed": not missing, "missing_files": missing}
    gg_path, paired_path = project_path(config["selection_gg_path"]), project_path(config["paired_splits_path"])
    missing = [str(p) for p in (gg_path, paired_path) if not p.is_file()]
    if not missing:
        gg = json.loads(gg_path.read_text(encoding="utf-8"))
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        if (gg.get("protocol") != config["protocol"] or gg.get("cache_version") != config["cache_version"]
                or len(gg["all"]) != 116 or len(set(gg["budgets"]["K19"]["pool"])) != 19
                or set(gg["all"]) != set(paired["U"])
                or set(paired["T_pair"]) != set(paired["U"]) - (set(paired["P_old"]) | set(paired["P_gg"]))):
            raise ValueError("GG选标产物预算或共同清单不符")
    result["gg_selection"] = {"completed": not missing, "missing_files": missing,
                              "elapsed_seconds": gg.get("elapsed_seconds") if not missing else None,
                              "k_elapsed_seconds": gg.get("k_elapsed_seconds") if not missing else None}
    rs_path = ROOT / "outputs/eval/rs/analysis.json"
    rs_missing = not rs_path.is_file()
    if not rs_missing:
        rs = json.loads(rs_path.read_text(encoding="utf-8"))
        old = json.loads(project_path(config["selection_path"]).read_text(encoding="utf-8"))
        if set(rs["pool"]) != set(old["budgets"]["K19"]["pool"]) or len(rs["importance"]) != 25:
            raise ValueError("RS不属于冻结旧19池或维数不符")
    result["rs_analysis"] = {"completed": not rs_missing, "missing_files": [str(rs_path)] if rs_missing else []}
    return result


def budget_info(train_root):
    result = []
    for folder in sorted(train_root.iterdir()) if train_root.is_dir() else []:
        if not folder.is_dir():
            continue
        for name in ("fit_info.json", "calibration.json", "config.json"):
            path = folder / name
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                costs = {k: v for k, v in value.items() if k in
                         ("elapsed_seconds", "python_traced_peak_bytes", "process_peak_memory_bytes", "support_vector_count", "fit_count", "calibration_fit_count")}
                if costs:
                    result.append({"folder": str(folder), "file": name, **costs})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="阶段三只读汇总；未完成格如实列出")
    parser.add_argument("--config", default="configs/stage3.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    reports = discover_reports(ROOT / "outputs/eval", config)
    result = summarize(config, reports)
    result["auxiliary_tasks"] = auxiliary_status(config)
    result["training_tasks"] = training_status(config, reports)
    result["budget_measurements"] = budget_info(ROOT / "outputs/train")
    result["pilot_budget_measurements"] = budget_info(ROOT / "outputs/pilot")
    output = new_output(args.output or f"outputs/summary_{date.today().isoformat()}")
    dump_json(output / "summary.json", result)
    lines = ["# 阶段三结果汇总", "", f"主要结果完成 {result['main_completed']}/144 格，未完成 {len(result['main_missing'])} 格。", "", "| 臂 | 完成 | E1均值 | E2均值 |", "| --- | ---: | ---: | ---: |"]
    for arm, value in result["arms"].items():
        e1, e2 = value["E1"]["mean"], value["E2"]["mean"]
        lines.append(f"| {arm} | {value['completed']}/12 | {'无' if e1 is None else f'{e1:.4f}'} | {'无' if e2 is None else f'{e2:.4f}'} |")
    lines += ["", result["interpretation"], "", "逐格、标准差/最差格、共同留出配对差、未完成清单和零算力错误结构见 summary.json。"]
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"只读汇总已写入：{output}，主要结果 {result['main_completed']}/144")


if __name__ == "__main__":
    main()
