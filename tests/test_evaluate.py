# -*- coding: utf-8 -*-
"""阶段三评估小测试：聚合/E2/KB/save_reports/read_predictions/checkpoint 校验/dry-run。

只用合成数据，不训练、不重估 BN、不读真实缓存。
"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate
from evaluate import (_check_checkpoint, _check_supervised_pool, _output_dir, _paired_splits,
                      _read_eval_folder, _stage3_info, _write_predictions, aggregate_records,
                      aggregate_units, infer_records, main, read_predictions, save_reports,
                      score_predictions)


def pu_records(bearing="KI01", condition="N15_M01_F10", label=2, count=20):
    return [{"record_key": f"{bearing}/{condition}_{bearing}_{i + 1}.mat::vibration_1",
             "unit_key": f"{bearing}/{condition}", "path": f"{condition}_{bearing}_{i + 1}.mat",
             "dataset": "paderborn", "channel": "vibration_1", "label": label,
             "src_fs": 64000, "source": "paderborn", "bearing_id": bearing,
             "condition": condition, "origin": "artificial", "severity": 1,
             "start": i, "stop": i + 1} for i in range(count)]


def pu_predictions(**kwargs):
    return [{**row, "probabilities": [0.1, 0.1, 0.7, 0.1], "window_count": 1}
            for row in pu_records(**kwargs)]


def cw_predictions(label=1, source="12k_DE", filename="B007_0.mat"):
    unit = f"{source}/{filename}"
    return [{"record_key": f"{unit}::{channel}", "unit_key": unit, "path": filename,
             "dataset": "cwru", "channel": channel, "label": label, "source": source,
             "bearing_id": None, "condition": None, "origin": None, "severity": None,
             "src_fs": 12000, "start": index, "stop": index + 1,
             "probabilities": np.eye(4)[label].tolist(), "window_count": 1}
            for index, channel in enumerate(("DE", "FE"))]


def paired_payload():
    """合法双池：P_old 与 P_gg 只部分重叠；配对清单由冻结 selection 函数生成。"""
    from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
    from selection import paired_splits
    u = sorted(f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS for condition in CONDITIONS)
    p_old, p_gg = u[:19], u[15:34]
    selection = {"protocol": "v8_stage3", "k_selected": 19, "all": u,
                 "budgets": {"K19": {"pool": p_old, "test": [x for x in u if x not in p_old]},
                             "K116": {"pool": u, "test": []}}}
    selection_gg = {"protocol": "v8_stage3", "k_selected": 19, "all": u,
                    "budgets": {"K19": {"pool": p_gg, "test": [x for x in u if x not in p_gg]},
                                "K116": {"pool": u, "test": []}}}
    return paired_splits(selection, selection_gg), selection, selection_gg


def checkpoint_payload(arm="joint"):
    from oneclass import load_config
    config = load_config("configs/stage3.json")
    return {"protocol": "v8_stage3", "arm": arm, "fold": 0, "seed": 0,
            "checkpoint_kind": "last", "epoch": 30, "checkpoint_epoch": 30, "last_epoch": 30,
            "run_kind": "formal", "cache_version": config["cache_version"], "repr_config": {},
            "source_split": {"train": [], "val": []},
            "scaler": {"mean": [0.0] * 25, "std": [1.0] * 25},
            "model": {"n_channels": 2, "n_classes": 4, "frontend": "spectral_v1"}}


class TinyModel(torch.nn.Module):
    def __init__(self, tuple_output=False, corrupt_bn=False, invalid=False):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1)
        self.head = torch.nn.Linear(256, 4)
        self.tuple_output, self.corrupt_bn, self.invalid = tuple_output, corrupt_bn, invalid
        self.grad_modes = []
        self.eval()

    def forward(self, inputs):
        assert not self.training and not self.bn.training and not torch.is_grad_enabled()
        self.grad_modes.append(torch.is_grad_enabled())
        logits = self.head(self.bn(inputs).flatten(1))
        if self.corrupt_bn:
            self.bn.running_mean.add_(1)
        if self.invalid:
            logits = logits * float("nan")
        return (logits, None) if self.tuple_output else logits


class AggregateTests(unittest.TestCase):
    def test_record_then_unit_equal_weight_not_window_weight(self):
        records = pu_records()
        probabilities, offset = [], 0
        for index, row in enumerate(records):
            count = 100 if index == 0 else 1
            row.update(start=offset, stop=offset + count)
            offset += count
            probabilities.extend([[1, 0, 0, 0] if index == 0 else [0, 0, 1, 0]] * count)
        rows = aggregate_records(records, np.asarray(probabilities))
        unit = aggregate_units(rows)[0]
        self.assertEqual((unit["record_count"], unit["window_count"]), (20, 119))
        np.testing.assert_allclose(unit["probabilities"], [0.05, 0, 0.95, 0])
        self.assertEqual(np.asarray(unit["probabilities"]).argmax(), 2)

    def test_ka08_exact_19_and_rejections(self):
        self.assertEqual(aggregate_units(pu_predictions(bearing="KA08", count=19, label=3))[0]["record_count"], 19)
        for kwargs in ({"bearing": "KA08", "count": 20}, {"count": 19}, {"count": 21}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aggregate_units(pu_predictions(**kwargs))
        valid = pu_predictions()
        for mutation in (lambda rows: rows.pop(),
                         lambda rows: rows[1].update(label=3),
                         lambda rows: rows[1].update(channel="phase_current_1"),
                         lambda rows: rows[1].update(probabilities=[1, 1, 1, 1])):
            rows = [dict(row) for row in valid]
            mutation(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate_units(rows)

    def test_cw_de_fe_and_source_metrics(self):
        rows = []
        for label in range(4):
            rows.extend(cw_predictions(label=label, filename=f"record_{label}.mat"))
        units = aggregate_units(rows)
        self.assertEqual(len(units), 4)
        self.assertEqual(units[0]["channels"], ["DE", "FE"])
        scores = score_predictions(rows)
        self.assertEqual(scores["source"]["E1"]["support"], [2, 2, 2, 2])
        self.assertEqual(scores["source_by_channel"]["DE"]["n"], 4)

    def test_e2_masks_probability_not_logits_ball_can_win_e1(self):
        with torch.no_grad():
            probabilities = torch.softmax(torch.tensor([[-8.0, -1.0, -3.0, -4.0]]), dim=1).numpy()[0]
        row = {**pu_predictions(count=1)[0], "probabilities": probabilities.tolist()}
        scores = score_predictions([row], pool_units=[])
        self.assertEqual((scores["E1"]["accuracy"], scores["E2"]["accuracy"], scores["P_ball"]), (0, 1, 1))
        self.assertEqual(scores["E1"]["n"], scores["E2"]["n"])
        # Ball 仍可被 E1 选中：Ball 概率最大时 E1 判 Ball，E2 分母不变。
        row2 = {**row, "probabilities": [0.1, 0.6, 0.2, 0.1]}
        scores2 = score_predictions([row2])
        self.assertEqual((scores2["E1"]["accuracy"], scores2["P_ball"]), (0.0, 1.0))


class ScoreAndReportTests(unittest.TestCase):
    def test_pu_macro_supported_classes_and_kb_exclusion(self):
        rows = []
        for truth, guesses in ((0, [0, 0, 0]), (2, [1]), (3, [3])):
            for guess in guesses:
                rows.append({**pu_predictions(count=1)[0], "label": truth,
                             "probabilities": np.eye(4)[guess].tolist()})
        kb = [{**pu_predictions(bearing=b, label=None, count=1)[0],
               "probabilities": np.eye(4)[g].tolist()} for b, g in (("KB23", 2), ("KB24", 2), ("KB27", 3))]
        scores = score_predictions(rows + kb, pool_units=["KI01/another"])
        self.assertEqual(scores["n"], 5)
        self.assertEqual(scores["E1"]["averaged_classes"], ["Normal", "IR", "OR"])
        self.assertAlmostEqual(scores["E1"]["macro_recall"], 2 / 3)
        self.assertEqual(scores["P_ball"], 0.2)
        self.assertEqual(scores["KB"]["n"], 3)
        self.assertEqual(scores["KB"]["by_bearing"]["KB27"]["official_dominant_reference"], 3)
        self.assertEqual(scores["groups"]["seen_unseen_bearing"]["seen"]["n"], 5)
        self.assertEqual(scores["groups"]["severity"]["1"]["n"], 5)

    def test_save_reports_pairing_splits_and_no_overwrite(self):
        rows = pu_predictions()
        units = aggregate_units(rows)
        t_old = [units[0]["unit_key"]]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "rule.json").write_text("{}", encoding="utf-8")
            metrics = save_reports(rows, units, output, pool_units=[],
                                   eval_info={"arm": "phys_ies", "evaluation_scope": "zero_shot_all_primary",
                                              "heldout_scope": "zero_shot", "transductive": False,
                                              "evaluation_splits": {"T_old": t_old, "T_pair": []}})
            self.assertIn("T_old", metrics["splits"])
            self.assertEqual(metrics["splits"]["T_old"]["unit_level"]["n"], 1)
            self.assertEqual(metrics["splits"]["T_pair"]["interpretation"], "no_heldout")
            self.assertEqual(metrics["splits"]["T_old"]["supervised_pool_overlap_unit_count"], 0)
            self.assertIsNone(metrics["D3"]["passed"])
            self.assertEqual(metrics["unit_level"]["protocol"] if "protocol" in metrics["unit_level"] else
                             "missing", "missing")
            with self.assertRaises(FileExistsError):
                save_reports([], [], output)
        for name in ("metrics.json", "predictions.csv", "unit_predictions.csv", "eval_info.json"):
            with tempfile.TemporaryDirectory() as folder:
                (Path(folder) / name).write_text("existing", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    save_reports([], [], Path(folder))

    def test_old_style_csv_allows_empty_record_interval(self):
        """v7 旧完整记录锚 predictions.csv 的 start/stop 为空，仍可读且不补窗。"""
        fields = ["record_key", "unit_key", "path", "channel", "start", "stop", "dataset",
                  "source", "bearing_id", "condition", "origin", "severity", "src_fs", "label",
                  "prob_Normal", "prob_Ball", "prob_IR", "prob_OR",
                  "prediction_E1", "prediction_E2", "window_count"]

        def values(index):
            return [f"K001/N15_M01_F10_K001_{index}.mat::vibration_1", "K001/N15_M01_F10",
                    f"K001/N15_M01_F10_K001_{index}.mat", "vibration_1", "", "", "paderborn",
                    "paderborn", "K001", "N15_M01_F10", "none", "0", "64000", "0",
                    "1.0", "0.0", "0.0", "0.0", "0", "0", "94"]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text(",".join(fields) + "\n"
                            + "\n".join(",".join(values(i + 1)) for i in range(20)) + "\n",
                            encoding="utf-8")
            rows = read_predictions(path)
            self.assertIsNone(rows[0]["start"])
            self.assertIsNone(rows[0]["stop"])
            self.assertEqual(rows[0]["window_count"], 94)
            self.assertEqual(aggregate_units(rows)[0]["record_count"], 20)
            # 其他整数列空值仍然报错，不静默填 None。
            broken = values(1)
            broken[fields.index("window_count")] = ""
            (Path(folder) / "broken.csv").write_text(
                ",".join(fields) + "\n" + ",".join(broken) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_predictions(Path(folder) / "broken.csv")

    def test_read_predictions_optional_evidence_columns_roundtrip(self):
        rows = pu_predictions(bearing="KB23", label=None, count=1)
        rows[0].update(logspec_window_count=93, e_ir=1.5, e_or=0.5, anomaly_mean=0.25,
                       normal_threshold_fraction=0.8, anomaly_quantiles=[0.1, 0.2, 0.3])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            _write_predictions(rows, path)
            back = read_predictions(path)
        self.assertEqual(back[0]["logspec_window_count"], 93)
        self.assertAlmostEqual(back[0]["e_ir"], 1.5)
        self.assertAlmostEqual(back[0]["e_or"], 0.5)
        self.assertAlmostEqual(back[0]["anomaly_mean"], 0.25)
        self.assertAlmostEqual(back[0]["normal_threshold_fraction"], 0.8)
        np.testing.assert_allclose(back[0]["anomaly_quantiles"], [0.1, 0.2, 0.3])

    def test_read_predictions_roundtrip_with_null_label(self):
        rows = pu_predictions(bearing="KB23", label=None, count=1) + cw_predictions()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            from evaluate import _write_predictions
            _write_predictions(rows, path)
            back = read_predictions(path)
        self.assertEqual(len(back), 3)
        self.assertIsNone(back[0]["label"])
        np.testing.assert_allclose(back[0]["probabilities"], [0.1, 0.1, 0.7, 0.1])
        self.assertEqual(back[1]["prediction_E1"], 1)
        self.assertEqual(back[1]["window_count"], 1)
        unit_rows = pu_predictions(count=1) + [pu_predictions(count=1)[0] | {"unit_key": "KI02/C"}]
        for row in unit_rows:
            row.update(record_count=20, channels=["vibration_1"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unit_predictions.csv"
            from evaluate import _write_predictions
            _write_predictions(unit_rows, path, unit=True)
            back = read_predictions(path, unit=True)
        self.assertEqual(back[0]["channels"], ["vibration_1"])
        self.assertEqual(back[0]["record_count"], 20)


class InferenceTests(unittest.TestCase):
    def test_inference_eval_no_grad_bn_unchanged_tuple_logits(self):
        feature = type("F", (), {"X": torch.randn(3, 1, 256),
                                 "records": pu_records(count=3)})()
        for tuple_output in (False, True):
            model = TinyModel(tuple_output=tuple_output)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            rows = infer_records(model, feature, batch_size=2, device="cpu")
            self.assertEqual(len(rows), 3)
            self.assertEqual(model.grad_modes, [False, False])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[key], value), key)
            for row in rows:
                self.assertAlmostEqual(sum(row["probabilities"]), 1, places=6)
        with self.assertRaisesRegex(RuntimeError, "BN"):
            infer_records(TinyModel(corrupt_bn=True), feature)
        with self.assertRaises(ValueError):
            infer_records(TinyModel(invalid=True), feature)
        with self.assertRaises(ValueError):
            infer_records(TinyModel(), feature, batch_size=0)

    def test_checkpoint_checks_reject_best_epochs_pilot_and_missing_scaler(self):
        from oneclass import load_config
        from preprocessing import build_repr_config
        config = load_config("configs/stage3.json")
        repr_config = build_repr_config()
        cache = type("C", (), {"version": config["cache_version"],
                               "repr_config": repr_config})()
        valid = {**checkpoint_payload(), "repr_config": repr_config}
        model_config, scaler = _check_checkpoint(valid, config, "joint", 0, 0, cache)
        self.assertEqual(scaler["std"], [1.0] * 25)
        for mutate, message in (
                (lambda c: {**c, "checkpoint_kind": "best"}, "last"),
                (lambda c: {**c, "epoch": 29}, "30"),
                (lambda c: {**c, "last_epoch": 31}, "30"),
                (lambda c: {k: v for k, v in c.items() if k != "scaler"}, "scaler"),
                (lambda c: {**c, "run_kind": "pilot"}, "正式"),
                (lambda c: {**c, "arm": "ce_K19"}, "折"),
                (lambda c: {k: v for k, v in c.items() if k != "source_split"}, "source_split"),
                (lambda c: {**c, "protocol": "v7_stage2"}, "v8_stage3"),
                (lambda c: {**c, "scaler": {"mean": [0.0] * 25, "std": [0.0] * 25}}, "std")):
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, message):
                _check_checkpoint(mutate(valid), config, "joint", 0, 0, cache)


class CliTests(unittest.TestCase):
    def test_dry_run_deep_and_fusion_print_plan_without_files(self):
        for argv in (["--arm", "joint", "--dataset", "paderborn", "--checkpoint", "missing.pt"],
                     ["--arm", "svm_phys", "--dataset", "cwru", "--fold", "0", "--seed", "0"],
                     ["--arm", "fusion_joint", "--fold", "0", "--seed", "0"]):
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                output = root / "outputs" / "out"
                buffer = StringIO()
                # 正式输出路径受 v8/outputs 约束，测试用 mock ROOT 指向临时目录。
                with patch.object(evaluate, "ROOT", root), \
                        patch.object(__import__("sys"), "argv",
                                     ["evaluate.py", "--config", "configs/stage3.json",
                                      "--output", str(output), *argv]), \
                        redirect_stdout(buffer):
                    main()
                plan = json.loads(buffer.getvalue())
                self.assertFalse(plan["execute"])
                self.assertFalse(output.exists())


class PairedSplitTests(unittest.TestCase):
    def _scopes(self, paired, selection, selection_gg):
        payload = {"paired_splits.json": paired, "selection.json": selection,
                   "selection_gg.json": selection_gg}
        from oneclass import load_config
        with patch.object(evaluate, "_read_json",
                          side_effect=lambda path: payload[Path(path).name]):
            return _paired_splits(load_config("configs/stage3.json"))

    def test_legal_pools_only_hold_out_their_own_pool(self):
        paired, selection, selection_gg = paired_payload()
        # T_old 含 P_gg 中不属于 P_old 的单元是合法的；只有 T_pair 避开两池并集。
        self.assertTrue(set(paired["T_old"]) & set(paired["P_gg"]))
        scopes = self._scopes(paired, selection, selection_gg)
        self.assertEqual((len(scopes["all_primary"]), len(scopes["T_old"]),
                          len(scopes["T_gg"]), len(scopes["T_pair"])),
                         (116, 97, 97, len(paired["T_pair"])))
        self.assertFalse(set(scopes["T_pair"]) & (set(paired["P_old"]) | set(paired["P_gg"])))

    def test_t_pair_overlapping_pool_or_changed_pool_rejected(self):
        paired, selection, selection_gg = paired_payload()
        for mutation in (lambda p: {**p, "T_pair": p["T_pair"] + [p["P_old"][0]]},
                         lambda p: {**p, "P_old": p["P_old"][:18]},
                         lambda p: {**p, "T_gg": p["T_old"]}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self._scopes(mutation(paired), selection, selection_gg)

    def test_selection_pool_size_must_be_19(self):
        paired, selection, selection_gg = paired_payload()
        broken = json.loads(json.dumps(selection_gg))
        broken["budgets"]["K19"]["pool"] = broken["budgets"]["K19"]["pool"][:18]
        with self.assertRaises(ValueError):
            self._scopes(paired, selection, broken)


class ScopeInfoTests(unittest.TestCase):
    def _info(self, arm):
        from oneclass import load_config
        from preprocessing import build_repr_config
        config = load_config("configs/stage3.json")
        cache = type("C", (), {"version": config["cache_version"],
                               "repr_config": build_repr_config(), "records": []})()
        paired, selection, selection_gg = paired_payload()
        payload = {"paired_splits.json": paired, "selection.json": selection,
                   "selection_gg.json": selection_gg}
        with patch.object(evaluate, "_read_json",
                          side_effect=lambda path: payload[Path(path).name]):
            return _stage3_info(config, arm, "paderborn", cache, pool_units=None, fold=0, seed=0)[0]

    def test_zero_pool_is_zero_shot_and_only_k116_has_no_heldout(self):
        self.assertEqual(self._info("joint")["heldout_scope"], "zero_shot")
        self.assertEqual(self._info("adabn")["heldout_scope"],
                         "transductive_labels_heldout_signals_seen")
        self.assertEqual(self._info("ce_K19")["heldout_scope"], "supervised_pool_outside")
        self.assertEqual(self._info("ce_K116")["heldout_scope"], "no_heldout")
        self.assertEqual(self._info("fusion_ce_K19_without_phys")["heldout_scope"],
                         "supervised_pool_outside")
        self.assertEqual(self._info("fusion_joint_without_normal")["heldout_scope"], "zero_shot")


class ProvenanceAndOutputTests(unittest.TestCase):
    def _provenance(self):
        return {"checkpoint": "outputs/train/joint_f0_s0/last.pt", "checkpoint_kind": "last",
                "epoch": 30, "last_epoch": 30, "repr_config": {},
                "scaler": {"mean": [0.0] * 25, "std": [1.0] * 25},
                "source_split": {"train": [], "val": []}, "supervised_pool_units": []}

    def test_check_supervised_pool_rejects_missing_none_and_wrong(self):
        from oneclass import load_config
        config = load_config("configs/stage3.json")
        _, selection, selection_gg = paired_payload()
        payload = {"selection.json": selection, "selection_gg.json": selection_gg}
        with patch.object(evaluate, "_read_json",
                          side_effect=lambda path: payload[Path(path).name]):
            self.assertEqual(_check_supervised_pool(
                {**checkpoint_payload(), "supervised_pool_units": []}, config, "joint"), [])
            for mutation in (lambda c: {k: v for k, v in c.items() if k != "supervised_pool_units"},
                             lambda c: {**c, "supervised_pool_units": None},
                             lambda c: {**c, "supervised_pool_units": ["X"]}):
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "监督池"):
                    _check_supervised_pool(mutation(checkpoint_payload()), config, "joint")

    def test_deep_eval_folder_requires_provenance_and_real_intervals(self):
        from oneclass import load_config
        config = load_config("configs/stage3.json")
        rows = pu_predictions(count=3)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            info_path = folder / "eval_info.json"
            info_path.write_text(json.dumps({
                "protocol": "v8_stage3", "arm": "joint", "fold": 0, "seed": 0,
                "dataset": "paderborn", "cache_version": config["cache_version"],
                "run_kind": "formal", "provenance": self._provenance()}), encoding="utf-8")
            (folder / "metrics.json").write_text("{}", encoding="utf-8")
            _write_predictions(rows, folder / "predictions.csv")
            (folder / "unit_predictions.csv").write_text("", encoding="utf-8")
            back, provenance = _read_eval_folder(folder, "joint", 0, 0, "paderborn", config)
            self.assertEqual(len(back), 3)
            self.assertEqual(provenance["scaler"], self._provenance()["scaler"])
            # provenance 缺失或监督池为 None 都不得绕过。
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info.pop("provenance")
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                _read_eval_folder(folder, "joint", 0, 0, "paderborn", config)
            info["provenance"] = {**self._provenance(), "supervised_pool_units": None}
            info_path.write_text(json.dumps(info), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "监督池"):
                _read_eval_folder(folder, "joint", 0, 0, "paderborn", config)
            # 旧样式空 start/stop 不能冒充带真实区间的深度报告。
            info["provenance"] = self._provenance()
            info_path.write_text(json.dumps(info), encoding="utf-8")
            fields = ["record_key", "unit_key", "path", "channel", "start", "stop", "dataset",
                      "source", "bearing_id", "condition", "origin", "severity", "src_fs", "label",
                      "prob_Normal", "prob_Ball", "prob_IR", "prob_OR",
                      "prediction_E1", "prediction_E2", "window_count"]
            with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                handle.write(",".join(fields) + "\n")
                for row in rows:
                    handle.write(",".join([row["record_key"], row["unit_key"], row["path"],
                                           row["channel"], "", "", "paderborn", "paderborn",
                                           row["bearing_id"], row["condition"], "artificial", "1",
                                           "64000", "2", "0.1", "0.1", "0.7", "0.1",
                                           "2", "2", "1"]) + "\n")
            with self.assertRaisesRegex(ValueError, "窗口区间"):
                _read_eval_folder(folder, "joint", 0, 0, "paderborn", config)

    def test_formal_outputs_must_stay_under_v8_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(evaluate, "ROOT", root):
                self.assertEqual(_output_dir(root / "outputs" / "eval" / "x"),
                                 (root / "outputs" / "eval" / "x").resolve())
                with self.assertRaisesRegex(ValueError, "outputs"):
                    _output_dir(root / "elsewhere")


class DeepRunTests(unittest.TestCase):
    def test_pu_deep_eval_records_provenance_and_own_tau(self):
        from model import MSRSABiLSTM
        from oneclass import load_config
        from preprocessing import build_repr_config
        config = load_config("configs/stage3.json")
        repr_config = build_repr_config()
        model = MSRSABiLSTM(n_channels=2, n_classes=4)
        checkpoint = {**checkpoint_payload("joint"), "repr_config": repr_config,
                      "state_dict": model.state_dict(),
                      "source_split": {"train": ["12k_DE/a.mat"], "val": ["12k_DE/b.mat"]},
                      "supervised_pool_units": []}
        pu_rows, cw_rows = pu_predictions(count=20), cw_predictions()
        paired, selection, selection_gg = paired_payload()
        payload = {"selection.json": selection, "selection_gg.json": selection_gg}
        target_cache = SimpleNamespace(version=config["cache_version"], repr_config=repr_config,
                                       records=pu_rows)
        source_cache = SimpleNamespace(version=config["cache_version"], repr_config=repr_config,
                                       records=cw_rows)

        def fake_subset(cache, unit_keys=None, channels=None, labeled=True):
            records = cache.records
            return SimpleNamespace(X=np.zeros((len(records), 1, 256), dtype=np.float32),
                                   phys25=np.zeros((len(records), 25), dtype=np.float32),
                                   records=records)

        def fake_infer(model, feature, batch_size=256, device=None, inputs=None):
            return pu_rows if feature.records[0]["dataset"] == "paderborn" else cw_rows

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            output = root / "outputs" / "eval" / "joint_f0_s0_paderborn"
            args = SimpleNamespace(arm="joint", fold=0, seed=0, dataset="paderborn",
                                   checkpoint=str(root / "train" / "last.pt"), output=str(output))
            with patch.object(evaluate, "ROOT", root), \
                    patch("features.load_cache",
                          side_effect=lambda dataset, cache_dir=None:
                              target_cache if dataset == "paderborn" else source_cache), \
                    patch("features.feature_subset", side_effect=fake_subset), \
                    patch("features.dual_input",
                          side_effect=lambda X, phys, scaler, logspec_only=False:
                              np.zeros((len(X), 2, 256), dtype=np.float32)), \
                    patch.object(evaluate, "infer_records", side_effect=fake_infer), \
                    patch.object(evaluate.torch, "load", return_value=checkpoint), \
                    patch.object(evaluate, "_read_json",
                                 side_effect=lambda path: payload[Path(path).name]), \
                    patch.object(evaluate, "_source_split",
                                 return_value={"train": ["12k_DE/a.mat"],
                                               "val": ["12k_DE/b.mat"]}), \
                    patch.object(evaluate, "_paired_splits", return_value=paired):
                evaluate._run_deep(config, args, execute=True)
            info = json.loads((output / "eval_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["provenance"]["checkpoint"],
                             str((root / "train" / "last.pt").resolve()))
            self.assertEqual(info["provenance"]["supervised_pool_units"], [])
            self.assertEqual(info["provenance"]["scaler"], checkpoint["scaler"])
            self.assertEqual(info["heldout_scope"], "zero_shot")
            self.assertEqual(info["uncertainty"]["scopes"]["main_pu"]["unit_count"], 1)
            self.assertEqual(info["uncertainty"]["scopes"]["kb"]["unit_count"], 0)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["D3"]["n"], 1)
            self.assertIsNone(metrics["D3"]["passed"])


class ShallowRunTests(unittest.TestCase):
    def test_svm_phys_pu_eval_uses_frozen_pool_and_own_tau(self):
        from oneclass import load_config
        config = load_config("configs/stage3.json")
        paired, selection, selection_gg = paired_payload()
        payload = {"selection.json": selection, "selection_gg.json": selection_gg}
        pu_rows, cw_rows = pu_predictions(count=20), cw_predictions()
        target_cache = SimpleNamespace(version=config["cache_version"], repr_config={},
                                       records=pu_rows, phys25=None)
        source_cache = SimpleNamespace(version=config["cache_version"], repr_config={},
                                       records=cw_rows, phys25=None)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            output = root / "outputs" / "eval" / "svm_phys_f0_s0_paderborn"
            args = SimpleNamespace(arm="svm_phys", fold=0, seed=0, dataset="paderborn",
                                   model_dir=str(root / "train" / "svm_phys_f0_s0"),
                                   output=str(output))
            with patch.object(evaluate, "ROOT", root), \
                    patch("features.load_cache",
                          side_effect=lambda dataset, cache_dir=None:
                              target_cache if dataset == "paderborn" else source_cache), \
                    patch("features.feature_subset", side_effect=lambda cache, *a, **k: cache), \
                    patch("shallow.predict_records",
                          side_effect=lambda cache, model_dir:
                              cw_rows if cache is source_cache else pu_rows), \
                    patch.object(evaluate, "_check_auxiliary_model", return_value={}), \
                    patch.object(evaluate, "_read_json",
                                 side_effect=lambda path: payload[Path(path).name]), \
                    patch.object(evaluate, "_source_split",
                                 return_value={"train": ["12k_DE/a.mat"],
                                               "val": ["12k_DE/b.mat"]}), \
                    patch.object(evaluate, "_paired_splits", return_value=paired):
                evaluate._run_shallow(config, args, execute=True)
            info = json.loads((output / "eval_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["arm"], "svm_phys")
            self.assertEqual(set(info["supervised_pool_units"]),
                             set(selection["budgets"]["K19"]["pool"]))
            self.assertEqual(info["heldout_scope"], "supervised_pool_outside")
            self.assertEqual(info["uncertainty"]["scopes"]["main_pu"]["unit_count"], 1)


if __name__ == "__main__":
    unittest.main()
