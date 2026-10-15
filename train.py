# -*- coding: utf-8 -*-
"""阶段三深度训练入口；默认只检查计划（dry-run），显式 --execute 才构造模型并训练。

固定矩阵见 configs/stage3.json 与 docs/阶段三执行计划.md §5/§10：
- 12 个输出臂中只有 9 个深度运行臂能送入训练循环；svm_phys 走 shallow.py，
  fusion_* 走 evaluate.py，误用本入口执行会直接报错；
- 96 格从头/ce 梯度训练 + 12 格 AdaBN 重估，共 12×12 主格；
- 非空输出目录拒绝覆盖，失败格用 _retry1 重跑；
- --pilot-epochs 只用于 P4 预算试跑：run_kind=pilot、输出在 outputs/pilot/，
  正式 checkpoint 依赖仍拒绝任何 pilot 基座（P4 同目录满轮 pilot 除外）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
from features import CACHE_VERSION, feature_subset, fit_scaler, load_cache
from model import MSRSABiLSTM
from preprocessing import build_repr_config
from transfer import DEEP_ARMS, DEVICE, load_checkpoint, seed_everything, steps_per_epoch, train_model

ROOT = Path(__file__).resolve().parent

#: 12 个主要输出臂（含 svm_phys 与两个 fusion，dry-run 可写计划，执行拒绝）。
OUTPUT_ARMS = ("joint", "joint_logspec", "mixup4src", "coral4src", "adabn",
               "ce_K19", "ce_K19_gg", "ce_K116", "ce_K19_logspec",
               "svm_phys", "fusion_joint", "fusion_ce_K19")

#: 从头臂（同 seed 同初始权重）；其余深度臂依赖同格基座 last.pt。
SCRATCH_ARMS = ("joint", "joint_logspec", "mixup4src", "coral4src")

#: 依赖基座：adabn 与全部 ce_* 臂固定到同格 joint / joint_logspec 的 last。
BASE_ARM = {"adabn": "joint", "ce_K19": "joint", "ce_K19_gg": "joint",
            "ce_K116": "joint", "ce_K19_logspec": "joint_logspec"}

#: 非深度臂的真实依赖（只查文件存在性，不拟合）：Normal 模型与两数据集深度评估报告。
NORMAL_FILES = ("model.joblib", "scaler.json", "config.json", "split.json", "calibration.json")
DEEP_EVAL_FILES = ("eval_info.json", "metrics.json", "predictions.csv", "unit_predictions.csv")
FUSION_BASE = {"fusion_joint": "joint", "fusion_ce_K19": "ce_K19"}

#: 阶段三 §5 冻结训练参数；配置偏离即报错。
FROZEN_PARAMETERS = {"batch_size": 64, "epochs": 30, "joint_lr": 0.001, "adapt_lr": 0.0001,
                     "lambda_coral": 0.2, "mixup_input_weight": 0.5,
                     "mixup_feature_weight": 0.5, "bn_momentum": 0.1}
FROZEN_OPTIMIZER = {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-08,
                    "weight_decay": 0, "scheduler": None}


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_frozen(config):
    """只核对阶段三 §5/§10 冻结值，不引入协议 schema。"""
    if (config["protocol"] != "v8_stage3" or config["status"] != "frozen"
            or config["plan_version"] != "v8_stage3_plan_v1"):
        raise ValueError("配置必须是正式冻结的 v8_stage3 计划")
    if (config["arms"] != list(OUTPUT_ARMS) or config["deep_arms"] != list(DEEP_ARMS)
            or config["folds"] != [0, 1, 2, 3] or config["seeds"] != [0, 1, 2]
            or config["budgets"] != [19, 116] or config["split_seed"] != 0
            or config["selection_rule"] != "last" or config["class_balance"] is not True):
        raise ValueError("矩阵、划分或末轮选择规则与阶段三冻结值不一致")
    if config["parameters"] != FROZEN_PARAMETERS or config["optimizer"] != FROZEN_OPTIMIZER:
        raise ValueError("训练参数或优化器设置与冻结值不一致")
    if (config["model"] != {"n_channels": 2, "n_classes": 4, "frontend": "spectral_v1"}
            or config["representation"] != "dual25_logspec512"
            or config["cache_version"] != CACHE_VERSION):
        raise ValueError("网络、表征或缓存口径必须为阶段三冻结双通道配置")


def _check_gg_budget(budgets, all_units, name, size):
    """新 GG 档位的池/测试必须不重不漏覆盖 116 主单元。"""
    budget = budgets.get(name) if isinstance(budgets, dict) else None
    if not isinstance(budget, dict) or "pool" not in budget or "test" not in budget:
        raise ValueError(f"新 GG 选标缺少 budgets.{name}.pool/test")
    pool, test = budget["pool"], budget["test"]
    if (len(pool) != size or len(set(pool)) != size or set(pool) - set(all_units)
            or len(test) != len(set(test)) or set(pool) & set(test)
            or set(pool) | set(test) != set(all_units)):
        raise ValueError(f"新 GG {name} 池/测试必须不重不漏覆盖 116 主单元")


def _load_pools(config):
    """只读导入旧池与新 GG 池；显式按臂映射，禁止 removeprefix 式解析。

    旧池读 v8 只读副本（selection_path），并核对与 v7 原始 selection.json 字节一致；
    新 selection_gg.json 严核协议/缓存/表征与 K19、K116 的不重不漏；
    配对清单 paired_splits.json 同为主要依赖，补集关系不符即报错。
    """
    copy_path = _project_path(config["selection_path"])
    old_path = _project_path(config["old_selection_path"])
    if not copy_path.is_file() or not old_path.is_file():
        raise FileNotFoundError(f"旧池只读副本或 v7 原始选标缺失：{copy_path} / {old_path}")
    if copy_path.read_bytes() != old_path.read_bytes():
        raise ValueError(f"v8 旧池副本与 v7 原始选标字节不一致，必须重跑准备：{copy_path}")
    old = json.loads(copy_path.read_text(encoding="utf-8"))
    if (old.get("protocol") != "v7_stage2" or old.get("budgets_status") != "frozen"
            or old.get("k_selected") != 19):
        raise ValueError(f"旧池必须为冻结的 v7_stage2 selection.json：{copy_path}")
    all_units = old["all"]
    primary = {f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS for condition in CONDITIONS}
    if len(all_units) != 116 or set(all_units) != primary:
        raise ValueError("旧池全集必须恰为排 KB 的 116 主单元")
    k19, k116 = old["budgets"]["K19"], old["budgets"]["K116"]
    for name, budget, size in (("K19", k19, 19), ("K116", k116, 116)):
        pool, test = budget["pool"], budget["test"]
        if (len(pool) != size or len(set(pool)) != size or set(pool) - set(all_units)
                or len(test) != len(set(test)) or set(pool) & set(test)
                or set(pool) | set(test) != set(all_units)):
            raise ValueError(f"旧 {name} 池/测试必须不重不漏覆盖 116 主单元")
    if not set(k19["pool"]) <= set(k116["pool"]):
        raise ValueError("K19 池必须是 K116 的子集")
    gg_path = _project_path(config["selection_gg_path"])
    gg = json.loads(gg_path.read_text(encoding="utf-8"))
    if (gg.get("protocol") != "v8_stage3" or gg.get("cache_version") != CACHE_VERSION
            or gg.get("repr_config") != build_repr_config()
            or set(gg.get("all", [])) != set(all_units)):
        raise ValueError(f"新 GG 池必须是同一 116 全集、缓存与表征口径一致的 v8_stage3：{gg_path}")
    for name, size in (("K19", 19), ("K116", 116)):
        _check_gg_budget(gg.get("budgets", {}), all_units, name, size)
    if not set(gg["budgets"]["K19"]["pool"]) <= set(gg["budgets"]["K116"]["pool"]):
        raise ValueError("新 GG K19 池必须是 K116 的子集")
    paired_path = _project_path(config["paired_splits_path"])
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    u = set(paired.get("U", []))
    p_old, p_gg = set(paired.get("P_old", [])), set(paired.get("P_gg", []))
    if (u != set(all_units) or p_old != set(k19["pool"])
            or p_gg != set(gg["budgets"]["K19"]["pool"])
            or set(paired.get("T_old", [])) != u - p_old
            or set(paired.get("T_gg", [])) != u - p_gg
            or set(paired.get("T_pair", [])) != u - (p_old | p_gg)
            or not set(paired.get("common_unseen_units", [])) <= u - (p_old | p_gg)):
        raise ValueError(f"配对清单与冻结新旧池不一致：{paired_path}")
    return {"all": list(all_units), "K19_old": list(k19["pool"]),
            "K116": list(k116["pool"]), "K19_gg": list(gg["budgets"]["K19"]["pool"])}


def _window_count(cache, units, channels=None):
    units = set(units)
    return sum(r["stop"] - r["start"] for r in cache.records
               if r["unit_key"] in units and (channels is None or r["channel"] in channels))


def _try(fn, missing):
    """dry-run 依赖探测：只有缺文件记 pending；缓存过期/口径不符直接报错，不静默跳过。"""
    try:
        return fn()
    except FileNotFoundError as error:
        missing.append(str(error))
        return None


def _auxiliary_missing(arm, fold, seed):
    """非深度臂按真实依赖检查 pending（只查文件，不拟合）。

    svm_phys 的依赖（源/目标缓存与旧池）已在 prepare_plan 统一检查；
    fusion_* 依赖同格 Normal 模型与两个数据集的深度评估报告。
    """
    if arm not in FUSION_BASE:
        return []
    missing = []
    normal_dir = _project_path(f"outputs/train/normal_f{fold}")
    missing.extend(f"缺 Normal 模型文件：{normal_dir / name}"
                   for name in NORMAL_FILES if not (normal_dir / name).is_file())
    base = FUSION_BASE[arm]
    for dataset in ("cwru", "paderborn"):
        folder = _project_path(f"outputs/eval/{base}_f{fold}_s{seed}_{dataset}")
        missing.extend(f"缺深度评估报告：{folder / name}"
                       for name in DEEP_EVAL_FILES if not (folder / name).is_file())
    return missing


def prepare_plan(args):
    """只读缓存元数据与冻结清单；不构造模型、优化器或特征训练副本。"""
    config_path = _project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _check_frozen(config)
    if args.arm not in OUTPUT_ARMS:
        raise ValueError(f"{args.arm} 不是阶段三输出臂")
    if args.arm not in config["arms"] or args.fold not in config["folds"] \
            or args.seed not in config["seeds"]:
        raise ValueError("arm、fold 或 seed 不在冻结矩阵内")
    if args.pilot_epochs is not None and (
            args.arm not in DEEP_ARMS
            or not 1 <= args.pilot_epochs <= config["parameters"]["epochs"]):
        raise ValueError(f"pilot_epochs 只能用于深度臂且为 1..{config['parameters']['epochs']}")
    run_kind = "pilot" if args.pilot_epochs is not None else "formal"
    missing = []
    pools = _try(lambda: _load_pools(config), missing)
    cache_dir = _project_path(args.cache_dir or config["cache_dir"])
    source = _try(lambda: load_cache("cwru", cache_dir), missing)
    target = _try(lambda: load_cache("paderborn", cache_dir), missing)
    from datasets import cwru
    source_split = None
    s_ref = None
    steps = None
    if source is not None:
        recordings = cwru.enumerate_recordings(
            _project_path(config["data_roots"]["cwru"]))
        source_units = {r.unit_key for r in recordings}
        if (len(source_units) != 153 or len(recordings) != 306
                or {r["unit_key"] for r in source.records} != source_units):
            raise ValueError("源域必须是排 028 的全部 153 文件，拒绝旧 v6 源集")
        source_split = cwru.split_source_units(recordings, split_seed=config["split_seed"])[args.fold]
        s_ref = _window_count(source, source_split["train"], ("DE",))
        steps = steps_per_epoch(s_ref, config["parameters"]["batch_size"])
    # 非深度臂（svm/fusion）没有梯度轮次，规划字段不冒充训练预算。
    is_deep = args.arm in DEEP_ARMS
    planned_epochs = None if not is_deep else (
        args.pilot_epochs if args.pilot_epochs is not None else config["parameters"]["epochs"])
    planned_steps = None if not is_deep or steps is None else steps * planned_epochs
    # 池名显式映射（阶段三 §5/§10）：深度 ce_* 与 transfer.POOL_ARM_MAP 一致；
    # svm_phys 用旧 K19 拟合，fusion_ce_K19 继承基座 ce_K19 的旧 K19，fusion_joint 无目标标注。
    pool_name = {"ce_K19": "K19_old", "ce_K19_gg": "K19_gg", "ce_K116": "K116",
                 "ce_K19_logspec": "K19_old",
                 "svm_phys": "K19_old", "fusion_ce_K19": "K19_old"}.get(args.arm)
    if pools is None:
        target_train = None
    elif args.arm == "adabn":
        target_train = list(pools["all"])  # 无标签 AdaBN 只用 U，均匀采样不访问真值
    elif pool_name is not None:
        target_train = list(pools[pool_name])
    else:
        target_train = []
    labeled = pool_name is not None
    base_arm = BASE_ARM.get(args.arm)
    # pilot 运行的基座默认取同目录（outputs/pilot）的满轮 pilot joint；正式仍取 outputs/train。
    base_root = "outputs/pilot" if run_kind == "pilot" else "outputs/train"
    checkpoint = None if base_arm is None else _project_path(
        args.source_checkpoint or f"{base_root}/{base_arm}_f{args.fold}_s{args.seed}/last.pt")
    if base_arm is None and args.source_checkpoint:
        raise ValueError(f"{args.arm} 必须从头训练（同 seed 同初始权重），不能传入源 checkpoint")
    if checkpoint and not checkpoint.is_file():
        missing.append(f"缺少基座 checkpoint：{checkpoint}")
    if args.arm not in DEEP_ARMS:
        # 非深度臂不送训练循环；dry-run 仍按真实依赖（Normal/深度报告）检查 pending。
        missing.extend(_auxiliary_missing(args.arm, args.fold, args.seed))
    prefix = {"pilot": "outputs/pilot", "formal": "outputs/train"}[run_kind] \
        if args.execute else "outputs/preflight"
    output = _project_path(args.output or f"{prefix}/{args.arm}_f{args.fold}_s{args.seed}")
    if args.execute and not output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("正式及pilot产物只允许写入v8/outputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"不覆盖非空输出目录，请另用 _retry1：{output}")
    config.update({"arm": args.arm, "fold": args.fold, "seed": args.seed,
                   "config_path": str(config_path), "cache_dir": str(cache_dir),
                   "cache_version": CACHE_VERSION, "repr_config": build_repr_config(),
                   "source_checkpoint": str(checkpoint) if checkpoint else None,
                   "initialization": "scratch" if base_arm is None else f"{base_arm}_last",
                   "run_kind": run_kind, "pilot_epochs": args.pilot_epochs,
                   "pool_name": pool_name, "budget": pool_name,
                   "target_train_units": target_train,
                   # 选标清单缺失时 pending，不猜测池内容；有清单的 ce_* 臂报告预算数。
                   "target_label_usage": len(target_train)
                       if labeled and target_train is not None else 0,
                   "source_split": source_split, "s_ref": s_ref,
                   "steps_per_epoch": steps,
                   "selected_checkpoint": "last.pt"})
    split = {"source": source_split, "fold": args.fold,
             "target": {"all": list(pools["all"]) if pools is not None else None,
                        "pool": list(pools[pool_name])
                            if pool_name is not None and pools is not None else None,
                        "train": target_train, "budget": pool_name,
                        "labeled": labeled if pools is not None else None}}
    plan = {"protocol": config["protocol"], "plan_version": config["plan_version"],
            "arm": args.arm, "fold": args.fold, "seed": args.seed,
            "mode": "execute" if args.execute else "dry_run", "run_kind": run_kind,
            "pilot_epochs": args.pilot_epochs,
            "status": "dependency_pending" if missing else "ready",
            "missing_dependencies": missing, "output": str(output),
            "representation": config["representation"], "model": config["model"],
            "parameters": config["parameters"], "class_balance": True,
            "logspec_only": args.arm in ("joint_logspec", "ce_K19_logspec"),
            "initialization": config["initialization"],
            "source_checkpoint": config["source_checkpoint"],
            "s_ref": s_ref, "steps_per_epoch": steps if is_deep else None,
            "num_samples_per_epoch": None if not is_deep or steps is None
                else steps * config["parameters"]["batch_size"],
            "planned_epochs": planned_epochs,
            "planned_steps": planned_steps,
            "planned_updates": None if not is_deep else
                (0 if args.arm == "adabn" else planned_steps),
            "source_file_count": 153,
            "source_train_unit_count": None if source_split is None else len(source_split["train"]),
            "source_validation_unit_count": None if source_split is None else len(source_split["val"]),
            "source_train_window_count": None if source is None or source_split is None
                else _window_count(source, source_split["train"]),
            "source_validation_window_count": None if source is None or source_split is None
                else _window_count(source, source_split["val"]),
            "target_train_unit_count": None if target_train is None else len(target_train),
            "target_train_window_count": None if target is None or not target_train
                else _window_count(target, target_train),
            # 非深度臂不自行梯度采样：fusion 继承基座预算，svm 按 shallow 的单元配额取窗。
            "target_sampling": "none" if not target_train else
                "uniform" if args.arm == "adabn" else
                "inherit_base" if args.arm in FUSION_BASE else
                "unit_quota" if args.arm == "svm_phys" else "balanced",
            "target_label_usage": config["target_label_usage"],
            "evaluation_regime": "transductive" if args.arm == "adabn" else
                "supervised_contains_training_pool" if labeled else "source_only",
            "selection_rule": "last", "criterion": config["criterion"],
            "paired_chain": config["paired_chain"], "split": split,
            "entry": "shallow.py" if args.arm == "svm_phys"
                else "evaluate.py" if args.arm.startswith("fusion_") else "train.py"}
    return config, split, plan, source, target


def _pilot_base_allowed(config):
    """P4 预算试跑例外：ce_* pilot 臂可用 outputs/pilot 下同目录的满轮 pilot 基座。

    轮数已由 validate_source_checkpoint 的 epoch/last_epoch 必须等于 30 保证；
    正式运行（run_kind=formal）永远拒绝任何 pilot 基座。
    """
    if config.get("run_kind") != "pilot":
        return False
    return "outputs/pilot/" in str(config.get("source_checkpoint", "")).replace("\\", "/")


def validate_source_checkpoint(payload, config):
    """后续臂只接受同折同 seed 同配置的正式 joint/joint_logspec 完整末轮。

    P4 例外：同为 pilot 的 ce_* 臂可复用 outputs/pilot 下已跑满 30 轮的 pilot 基座。
    """
    epochs = config["parameters"]["epochs"]
    base = BASE_ARM[config["arm"]]
    required = {"protocol": "v8_stage3", "arm": base,
                "fold": config["fold"], "seed": config["seed"],
                "checkpoint_kind": "last", "epoch": epochs, "checkpoint_epoch": epochs,
                "last_epoch": epochs, "cache_version": config["cache_version"],
                "repr_config": build_repr_config(), "model": config["model"],
                "representation": config["representation"],
                "source_split": config["source_split"], "s_ref": config["s_ref"],
                "steps_per_epoch": config["steps_per_epoch"],
                "steps": config["steps_per_epoch"] * epochs,
                "updates": config["steps_per_epoch"] * epochs,
                "selection_rule": "last", "class_balance": True,
                "logspec_only": base == "joint_logspec", "supervised_pool_units": []}
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"源 checkpoint 的 {key} 不匹配，必须用同折同种子的 v8 {base} last")
    if payload.get("run_kind") != "formal" and not _pilot_base_allowed(config):
        raise ValueError(f"源 checkpoint 的 run_kind={payload.get('run_kind')!r} 不是正式运行；"
                         f"只有 P4 pilot 臂能复用 outputs/pilot 下同目录的满轮 pilot 基座")
    if payload.get("parameters") != config["parameters"]:
        raise ValueError("源 checkpoint 训练参数必须与冻结配置一致")
    if not isinstance(payload.get("scaler"), dict) or len(payload["scaler"].get("mean", [])) != 25:
        raise ValueError("源 checkpoint 必须携带 25 维源训练 scaler")


def build_parser():
    parser = argparse.ArgumentParser(description="阶段三深度训练，默认仅 dry-run（无模型/无训练）")
    parser.add_argument("--arm", choices=OUTPUT_ARMS, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/stage3.json")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--source-checkpoint", default=None,
                        help="默认读取同格基座 outputs/train/<base>_f{fold}_s{seed}/last.pt"
                             "（pilot 运行默认 outputs/pilot/）")
    parser.add_argument("--pilot-epochs", type=int, default=None,
                        help="P4 预算试跑：限定轮数、run_kind=pilot、输出到 outputs/pilot/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只检查并写 dry_run.json，不构造模型")
    mode.add_argument("--execute", action="store_true", help="须获训练授权后才使用")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config, split, plan, source, target = prepare_plan(args)
    output = Path(plan["output"])
    if not args.execute:
        output.mkdir(parents=True, exist_ok=True)
        _dump_json(output / "dry_run.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if args.arm not in DEEP_ARMS:
        raise ValueError(f"{args.arm} 不是深度运行臂，拒绝送入训练循环："
                         f"svm_phys 请用 shallow.py，fusion_* 请用 evaluate.py")
    if plan["status"] != "ready":
        raise FileNotFoundError(f"依赖缺失，不能执行：{plan['missing_dependencies']}")
    if plan["run_kind"] == "pilot" and not output.resolve().is_relative_to((ROOT / "outputs/pilot").resolve()):
        raise ValueError("pilot 试跑产物必须写到 outputs/pilot/ 下，不冒充正式格")
    # 在模型、优化器和训练前固定所有划分及配置；非空目录在 prepare_plan 已拒绝。
    output.mkdir(parents=True, exist_ok=True)
    _dump_json(output / "split.json", split)
    seed_everything(args.seed)
    model = MSRSABiLSTM(n_channels=config["model"]["n_channels"],
                        n_classes=config["model"]["n_classes"]).to(DEVICE)
    payload = None
    if config["initialization"] != "scratch":
        # 同格基座：默认只接受同折同 seed 的正式 joint/joint_logspec last（P4 pilot 例外见校验）。
        payload = load_checkpoint(plan["source_checkpoint"])
        validate_source_checkpoint(payload, config)
        model.load_state_dict(payload["state_dict"])
    train_source = feature_subset(source, split["source"]["train"])
    val_source = feature_subset(source, split["source"]["val"])
    if payload is not None:
        # 有基座的臂（ce_*/adabn）复用基座拟合的同一份缩放器，保证输入口径一致。
        scaler = payload["scaler"]
    else:
        # 从头臂在本折源训练窗口拟合逐维均值/总体标准差（§3.3）。
        scaler = fit_scaler(train_source.phys25.numpy())
    train_source.scaler = scaler
    val_source.scaler = scaler
    train_target = None
    if split["target"]["train"]:
        train_target = feature_subset(target, split["target"]["train"],
                                      labeled=split["target"]["labeled"])
        train_target.scaler = scaler
    _dump_json(output / "config.json", {**config, "scaler": scaler})
    return train_model(model, train_source, train_target, val_source, args.arm,
                       config, output, plan["s_ref"], args.seed,
                       run_kind=plan["run_kind"], pilot_epochs=args.pilot_epochs)


if __name__ == "__main__":
    main()
