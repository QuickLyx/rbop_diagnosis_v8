# -*- coding: utf-8 -*-
"""四个源外层折各一份 Normal 单类证据；不使用目标标签或目标缩放。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import joblib
import numpy as np
from scipy.special import expit
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path="configs/stage3.json"):
    config = json.loads(project_path(path).read_text(encoding="utf-8"))
    if config["protocol"] != "v8_stage3" or config["status"] != "frozen":
        raise ValueError("必须使用已冻结的阶段三配置")
    return config


def new_output(path):
    output = project_path(path).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("产物只允许写入 v8/outputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"不覆盖非空产物目录，请使用 _retry1：{output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def peak_memory_bytes():
    """操作系统进程历史峰值；一次拟合若需独立峰值应启动独立进程。"""
    import sys
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged_pool",
                "paged_pool", "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.peak_working_set)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def source_split(cache, fold, config):
    from datasets.cwru import split_source_units
    if fold not in config["folds"]:
        raise ValueError("源折不在冻结矩阵内")
    return split_source_units([SimpleNamespace(**row) for row in cache.records], config["split_seed"])[fold]


def normal_evidence(anomaly, threshold, scale):
    scores = np.asarray(anomaly, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("异常分、阈值及尺度必须有限，尺度必须为正")
    r = expit((threshold - scores) / scale)
    return np.column_stack((r, (1-r)/3, (1-r)/3, (1-r)/3))


def fit_normal(source_cache, fold, output, config):
    from features import feature_subset, fit_scaler, scale_phys
    split = source_split(source_cache, fold, config)
    train = feature_subset(source_cache, split["train"])
    val = feature_subset(source_cache, split["val"])
    scaler = fit_scaler(train.phys25)
    x_train = scale_phys(train.phys25, scaler)[np.asarray(train.y) == 0]
    x_val = scale_phys(val.phys25, scaler)[np.asarray(val.y) == 0]
    if not len(x_train) or not len(x_val):
        raise ValueError("Normal 源训练/验证窗口为空")
    parameters = config["normal"]
    expected = {"kernel": "rbf", "nu": 0.05, "gamma": 1/25, "tol": 1e-3,
                "threshold_quantile": 0.99, "scale_floor": 1e-6, "quantile_method": "linear", "seed": 0}
    if parameters != expected:
        raise ValueError("Normal 配方偏离冻结计划")
    output = new_output(output)
    start = time.perf_counter()
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma=1/25, tol=1e-3).fit(x_train)
    if model.fit_status_ != 0:
        raise RuntimeError("Normal 拟合未成功收敛，停止该折")
    a_train = -model.decision_function(x_train)
    a_val = -model.decision_function(x_val)
    if not np.isfinite(a_train).all() or not np.isfinite(a_val).all():
        raise ValueError("单类拟合产生非有限异常分，停止该折")
    threshold = float(np.quantile(a_val, 0.99, method="linear"))
    scale = max(float(np.quantile(a_train, 0.75, method="linear") - np.quantile(a_train, 0.25, method="linear")), 1e-6)
    joblib.dump(model, output / "model.joblib")
    dump_json(output / "scaler.json", scaler)
    dump_json(output / "config.json", {**config, "arm": "normal", "fold": fold, "seed": 0,
        "run_kind": "pilot" if output.resolve().is_relative_to((ROOT / "outputs/pilot").resolve()) else "formal"})
    dump_json(output / "split.json", split)
    dump_json(output / "calibration.json", {
        "t": threshold, "s": scale, "anomaly_direction": "越大越异常", "quantile_method": "linear",
        "train_normal_units": sorted({r["unit_key"] for r in train.records if r["label"] == 0}),
        "validation_normal_units": sorted({r["unit_key"] for r in val.records if r["label"] == 0}),
        "train_windows": len(x_train), "validation_windows": len(x_val),
        "source_validation_window_normal_fraction": float(np.mean(a_val <= threshold)),
        "elapsed_seconds": time.perf_counter() - start, "process_peak_memory_bytes": peak_memory_bytes(),
        "memory_note": "操作系统整个进程历史峰值，非该算法独占增量",
        "note": "一个验证Normal文件的两个通道；窗口经验分位不保证目标误报率，r不是真实后验"})
    return output


def predict_records(cache, model_dir):
    from features import scale_phys
    from evaluate import aggregate_records
    model_dir = project_path(model_dir)
    required = ("model.joblib", "scaler.json", "config.json", "split.json", "calibration.json")
    if any(not (model_dir / name).is_file() for name in required):
        raise FileNotFoundError(f"Normal 产物不完整：{model_dir}")
    metadata = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "v8_stage3" or metadata.get("arm") != "normal":
        raise ValueError("不是阶段三 Normal 模型")
    model = joblib.load(model_dir / "model.joblib")
    scaler = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    scores = np.empty(len(cache.phys25), dtype=np.float64)
    for start in range(0, len(scores), 4096):
        scores[start:start+4096] = -model.decision_function(scale_phys(cache.phys25[start:start+4096], scaler))
    rows = aggregate_records(cache.records, normal_evidence(scores, calibration["t"], calibration["s"]))
    for row in rows:
        a = scores[row["start"]:row["stop"]]
        row["anomaly_mean"] = float(a.mean())
        row["normal_threshold_fraction"] = float(np.mean(a <= calibration["t"]))
        row["anomaly_quantiles"] = np.quantile(a, [0.05, 0.5, 0.95], method="linear").tolist()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="源折 Normal 单类模型，默认仅检查依赖")
    parser.add_argument("--config", default="configs/stage3.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.fold not in config["folds"]:
        raise ValueError("fold 不在冻结矩阵中")
    if not args.execute:
        print(json.dumps({"arm": "normal", "fold": args.fold, "seed": 0, "fit": False,
                          "cache_dir": config["cache_dir"], "parameters": config["normal"]}, ensure_ascii=False))
        return
    from features import load_cache
    source = load_cache("cwru", project_path(config["cache_dir"]))
    fit_normal(source, args.fold, args.output or f"outputs/train/normal_f{args.fold}", config)


if __name__ == "__main__":
    main()
