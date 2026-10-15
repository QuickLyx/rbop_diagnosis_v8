# -*- coding: utf-8 -*-
"""只检查 train.py 的 144 主格计划、臂映射与执行防护；不触缓存、不构造模型。"""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from bearing_facts import CONDITIONS, PRIMARY_BEARINGS
from features import CACHE_VERSION, build_repr_config
import train


@contextmanager
def isolated_missing_dependencies():
    """合成缺缓存/缺池环境：测试不读真实 P2 产物（显式有依赖用例自行覆盖）。"""
    with mock.patch.object(train, "_load_pools",
                           side_effect=FileNotFoundError("合成测试：缺选标清单")), \
         mock.patch.object(train, "load_cache",
                           side_effect=FileNotFoundError("合成测试：缺缓存")):
        yield


@contextmanager
def forbidden_training():
    """禁止真实模型/优化器/训练循环：前置检查一旦回归，测试以断言失败而非真实训练结束。"""
    with mock.patch.object(train, "MSRSABiLSTM",
                           side_effect=AssertionError("测试禁止构造模型")), \
         mock.patch.object(train, "train_model",
                           side_effect=AssertionError("测试禁止真实训练")), \
         mock.patch("transfer.torch.optim.Adam",
                    side_effect=AssertionError("测试禁止构造优化器")):
        yield


class PlanGridTests(unittest.TestCase):
    """144 主格 = 12 输出臂 × 4 折 × 3 seed；96 梯度 + 12 AdaBN 由臂清单推出。"""

    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (Path(train.ROOT) / "configs/stage3.json").read_text(encoding="utf-8"))

    def test_frozen_grid_counts(self):
        self.assertEqual(len(self.config["arms"]), 12)
        self.assertEqual(len(self.config["deep_arms"]), 9)
        self.assertEqual(len(self.config["folds"]) * len(self.config["seeds"])
                         * len(self.config["arms"]), 144)
        gradient_arms = [arm for arm in self.config["deep_arms"] if arm != "adabn"]
        self.assertEqual(len(gradient_arms) * 12, 96)
        self.assertEqual(self.config["counts"]["gradient_runs"], 96)
        self.assertEqual(self.config["counts"]["adabn_runs"], 12)
        self.assertEqual(self.config["counts"]["main_outputs"], 144)
        # train.py 的臂清单必须与冻结配置逐项一致。
        self.assertEqual(tuple(self.config["arms"]), train.OUTPUT_ARMS)
        self.assertEqual(tuple(self.config["deep_arms"]), train.DEEP_ARMS if hasattr(
            train, "DEEP_ARMS") else tuple(train.DEEP_ARMS))

    def test_deep_arm_sets_match_transfer(self):
        import transfer
        self.assertEqual(set(self.config["deep_arms"]), set(transfer.DEEP_ARMS))
        self.assertEqual(set(self.config["arms"]) - set(self.config["deep_arms"]),
                         {"svm_phys", "fusion_joint", "fusion_ce_K19"})


class DryRunPlanTests(unittest.TestCase):
    """无缓存环境下逐臂 dry-run：全部 pending、不构造模型、入口映射正确。"""

    def _plan(self, arm, extra=()):
        with tempfile.TemporaryDirectory() as directory, \
                isolated_missing_dependencies(), forbidden_training():
            output = Path(directory) / arm
            with redirect_stdout(io.StringIO()):
                plan = train.main(["--arm", arm, "--fold", "0", "--seed", "0",
                                   "--output", str(output), *extra])
            files = sorted(path.name for path in output.iterdir())
        return plan, files

    def test_all_output_arms_write_pending_plan_without_model(self):
        entries = {"svm_phys": "shallow.py", "fusion_joint": "evaluate.py",
                   "fusion_ce_K19": "evaluate.py"}
        epochs = json.loads((Path(train.ROOT) / "configs/stage3.json").read_text(
            encoding="utf-8"))["parameters"]["epochs"]
        for arm in train.OUTPUT_ARMS:
            with self.subTest(arm=arm):
                plan, files = self._plan(arm)
                self.assertEqual(plan["mode"], "dry_run")
                self.assertEqual(plan["status"], "dependency_pending")
                self.assertEqual(plan["run_kind"], "formal")
                self.assertEqual(plan["entry"], entries.get(arm, "train.py"))
                self.assertEqual(files, ["dry_run.json"])
                self.assertNotIn("非深度臂不走训练循环", plan["missing_dependencies"])
                # 依赖缺失（含选标清单）时 pending，预算一律报 0，不猜测池内容。
                self.assertEqual(plan["target_label_usage"], 0)
                if arm in ("svm_phys", "fusion_joint", "fusion_ce_K19"):
                    # 非深度臂没有梯度轮次，计划字段不得冒充训练预算。
                    self.assertIsNone(plan["planned_epochs"])
                    self.assertIsNone(plan["planned_steps"])
                    self.assertIsNone(plan["planned_updates"])
                    self.assertIsNone(plan["num_samples_per_epoch"])
                    self.assertIsNone(plan["steps_per_epoch"])
                else:
                    self.assertEqual(plan["planned_epochs"], epochs)
                    self.assertEqual(plan["planned_updates"], 0 if arm == "adabn"
                                     else plan["planned_steps"])
                self.assertEqual(plan["selection_rule"], "last")

    def test_all_144_cells_preflight_without_model_or_optimizer(self):
        rows = []
        with tempfile.TemporaryDirectory() as directory, \
                isolated_missing_dependencies(), forbidden_training():
            for fold in range(4):
                for seed in range(3):
                    for arm in train.OUTPUT_ARMS:
                        args = train.build_parser().parse_args(["--arm", arm, "--fold", str(fold),
                            "--seed", str(seed), "--dry-run", "--output", str(Path(directory)/f"{arm}_{fold}_{seed}")])
                        _, _, plan, _, _ = train.prepare_plan(args)
                        rows.append((plan["arm"], plan["fold"], plan["seed"]))
                        self.assertEqual(plan["status"], "dependency_pending")
        self.assertEqual(len(set(rows)), 144)

    def test_pool_budget_mapping_is_explicit(self):
        # 显式池名映射：不做 removeprefix 式解析（ce_K19_gg ≠ K19）。
        for arm, budget in (("ce_K19", "K19_old"), ("ce_K19_gg", "K19_gg"),
                            ("ce_K116", "K116"), ("ce_K19_logspec", "K19_old"),
                            ("svm_phys", "K19_old"), ("fusion_ce_K19", "K19_old"),
                            ("joint", None), ("adabn", None), ("fusion_joint", None)):
            with self.subTest(arm=arm):
                plan, _ = self._plan(arm)
                self.assertEqual(plan["split"]["target"]["budget"], budget)

    def test_logspec_only_flags_follow_input_ablation_arms(self):
        for arm, expected in (("joint_logspec", True), ("ce_K19_logspec", True),
                              ("joint", False), ("ce_K19", False), ("mixup4src", False)):
            with self.subTest(arm=arm):
                plan, _ = self._plan(arm)
                self.assertEqual(plan["logspec_only"], expected)

    def test_initialization_follows_base_arm(self):
        for arm, init in (("joint", "scratch"), ("mixup4src", "scratch"),
                          ("coral4src", "scratch"), ("joint_logspec", "scratch"),
                          ("adabn", "joint_last"), ("ce_K19", "joint_last"),
                          ("ce_K19_gg", "joint_last"), ("ce_K116", "joint_last"),
                          ("ce_K19_logspec", "joint_logspec_last")):
            with self.subTest(arm=arm):
                plan, _ = self._plan(arm)
                self.assertEqual(plan["initialization"], init)

    def test_pilot_flags_only_for_deep_arms(self):
        epochs = json.loads((Path(train.ROOT) / "configs/stage3.json").read_text(
            encoding="utf-8"))["parameters"]["epochs"]
        plan, _ = self._plan("joint", ("--pilot-epochs", "1"))
        self.assertEqual(plan["run_kind"], "pilot")
        self.assertEqual(plan["planned_epochs"], 1)
        self.assertEqual(plan["planned_updates"], plan["steps_per_epoch"])
        with self.assertRaises(ValueError):
            self._plan("joint", ("--pilot-epochs", "0"))
        with self.assertRaises(ValueError):
            self._plan("svm_phys", ("--pilot-epochs", "1"))
        # pilot 基座默认取同目录 outputs/pilot；正式仍取 outputs/train。
        pilot_plan, _ = self._plan("ce_K19", ("--pilot-epochs", str(epochs)))
        self.assertIn("/pilot/joint_f0_s0/last.pt",
                      pilot_plan["source_checkpoint"].replace("\\", "/"))
        formal_plan, _ = self._plan("ce_K19")
        self.assertIn("/train/joint_f0_s0/last.pt",
                      formal_plan["source_checkpoint"].replace("\\", "/"))

    def test_pilot_epochs_upper_bound_checked_early(self):
        # 边界在 prepare_plan 提前拒绝，不等到 train_model（30 轮满轮 pilot 合法）。
        config = json.loads((Path(train.ROOT) / "configs/stage3.json").read_text(encoding="utf-8"))
        epochs = config["parameters"]["epochs"]
        plan, _ = self._plan("joint", ("--pilot-epochs", str(epochs)))
        self.assertEqual(plan["planned_epochs"], epochs)
        with self.assertRaises(ValueError):
            self._plan("joint", ("--pilot-epochs", str(epochs + 1)))

    def test_fusion_pending_follows_real_dependencies_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(train, "ROOT", Path(directory)):
            missing = train._auxiliary_missing("fusion_joint", 0, 0)
            self.assertTrue(missing)  # 空根目录：Normal 模型与两个深度评估报告都缺
            normal = Path(directory) / "outputs/train/normal_f0"
            normal.mkdir(parents=True)
            for name in train.NORMAL_FILES:
                (normal / name).write_text("{}", encoding="utf-8")
            for dataset in ("cwru", "paderborn"):
                folder = Path(directory) / f"outputs/eval/joint_f0_s0_{dataset}"
                folder.mkdir(parents=True)
                for name in train.DEEP_EVAL_FILES:
                    (folder / name).write_text("", encoding="utf-8")
            self.assertEqual(train._auxiliary_missing("fusion_joint", 0, 0), [])
            self.assertEqual(train._auxiliary_missing("svm_phys", 0, 0), [])

    def test_try_only_pends_on_missing_files(self):
        def boom(error):
            def raiser():
                raise error
            return raiser

        missing = []
        self.assertIsNone(train._try(boom(FileNotFoundError("缺缓存")), missing))
        self.assertEqual(len(missing), 1)
        # 过期缓存/口径错误必须直接报错，不作为 pending 静默跳过。
        with self.assertRaises(ValueError):
            train._try(boom(ValueError("缓存过期")), [])
        with self.assertRaises(KeyError):
            train._try(boom(KeyError("缺单元")), [])

    def test_output_rejects_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
                isolated_missing_dependencies(), forbidden_training():
            output = Path(directory) / "joint"
            with redirect_stdout(io.StringIO()):
                train.main(["--arm", "joint", "--fold", "0", "--seed", "0",
                            "--output", str(output)])
            (output / "keep.txt").write_text("占位", encoding="utf-8")
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                train.main(["--arm", "joint", "--fold", "0", "--seed", "0",
                            "--output", str(output)])


class PoolLoadingTests(unittest.TestCase):
    """_load_pools：旧池读 v8 只读副本并核 v7 字节，新 GG 严核版本与不重不漏。"""

    def _selection_payload(self, protocol, offset=0, cache_version=None):
        units = [f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS
                 for condition in CONDITIONS]
        self.assertEqual(len(units), 116)
        pool = units[offset:offset + 19]
        payload = {"protocol": protocol, "all": units, "k_selected": 19,
                   "budgets": {"K19": {"pool": sorted(pool),
                                        "test": sorted(set(units) - set(pool))},
                               "K116": {"pool": sorted(units), "test": []}}}
        if protocol == "v7_stage2":
            payload["budgets_status"] = "frozen"
        else:
            payload["cache_version"] = cache_version
            payload["repr_config"] = build_repr_config()
        return payload

    def _write_fixtures(self, root):
        old = self._selection_payload("v7_stage2")
        gg = self._selection_payload("v8_stage3", offset=19, cache_version=CACHE_VERSION)
        old_text = json.dumps(old, ensure_ascii=False)
        (root / "v7").mkdir()
        (root / "v7/selection.json").write_text(old_text, encoding="utf-8")
        (root / "outputs/selection").mkdir(parents=True)
        (root / "outputs/selection/selection.json").write_text(old_text, encoding="utf-8")
        (root / "outputs/selection/selection_gg.json").write_text(
            json.dumps(gg, ensure_ascii=False), encoding="utf-8")
        u = set(old["all"])
        p_old, p_gg = set(old["budgets"]["K19"]["pool"]), set(gg["budgets"]["K19"]["pool"])
        paired = {"U": sorted(u), "P_old": sorted(p_old), "P_gg": sorted(p_gg),
                  "T_old": sorted(u - p_old), "T_gg": sorted(u - p_gg),
                  "T_pair": sorted(u - (p_old | p_gg)), "common_unseen_units": []}
        (root / "outputs/selection/paired_splits.json").write_text(
            json.dumps(paired, ensure_ascii=False), encoding="utf-8")
        return {"selection_path": "outputs/selection/selection.json",
                "old_selection_path": "v7/selection.json",
                "selection_gg_path": "outputs/selection/selection_gg.json",
                "paired_splits_path": "outputs/selection/paired_splits.json"}

    def test_reads_copy_and_validates_gg(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(train, "ROOT", Path(directory)):
            config = self._write_fixtures(Path(directory))
            pools = train._load_pools(config)
            self.assertEqual(len(pools["K19_old"]), 19)
            self.assertEqual(len(pools["K19_gg"]), 19)
            self.assertEqual(len(pools["K116"]), 116)

    def test_rejects_byte_mismatch_and_stale_gg(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(train, "ROOT", Path(directory)):
            root = Path(directory)
            config = self._write_fixtures(root)
            copy_path = root / "outputs/selection/selection.json"
            original = copy_path.read_bytes()
            copy_path.write_text("{}已改", encoding="utf-8")
            with self.assertRaises(ValueError):
                train._load_pools(config)
            copy_path.write_bytes(original)
            gg_path = root / "outputs/selection/selection_gg.json"
            gg = json.loads(gg_path.read_text(encoding="utf-8"))
            gg_path.write_text(json.dumps({**gg, "cache_version": "v7_logspec512_12k_drop_ka08_v1"}),
                               encoding="utf-8")
            with self.assertRaises(ValueError):
                train._load_pools(config)
            gg_path.write_text(json.dumps({**gg, "budgets": {**gg["budgets"], "K19": {
                "pool": gg["budgets"]["K19"]["pool"] * 2, "test": []}}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                train._load_pools(config)

    def test_missing_paired_splits_is_pending_dependency(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(train, "ROOT", Path(directory)):
            config = self._write_fixtures(Path(directory))
            (Path(directory) / "outputs/selection/paired_splits.json").unlink()
            with self.assertRaises(FileNotFoundError):
                train._load_pools(config)


class ExecuteGuardTests(unittest.TestCase):
    """执行防护：非深度臂拒绝送训练循环；依赖缺失不得构造模型或进入训练。"""

    @contextmanager
    def _isolated_execute(self):
        """execute 类测试统一环境：强制缺依赖并禁止任何真实模型/训练构造。"""
        config_text = (Path(train.ROOT) / "configs/stage3.json").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/stage3.json").write_text(config_text, encoding="utf-8")
            with mock.patch.object(train, "ROOT", root), \
                    isolated_missing_dependencies(), forbidden_training():
                yield

    def test_execute_rejects_shallow_and_fusion_arms(self):
        for arm, entry in (("svm_phys", "shallow.py"), ("fusion_joint", "evaluate.py"),
                           ("fusion_ce_K19", "evaluate.py")):
            with self.subTest(arm=arm), self._isolated_execute(), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError) as caught:
                    train.main(["--arm", arm, "--fold", "0", "--seed", "0",
                                "--output", "outputs/preflight/guard_test", "--execute"])
                self.assertIn(entry, str(caught.exception))

    def test_execute_blocks_on_missing_dependencies(self):
        # 无缓存/无基座时 --execute 必须在构造模型前失败；环境强制缺依赖，绝不真实训练。
        for arm in ("joint", "mixup4src", "adabn", "ce_K19"):
            with self.subTest(arm=arm), self._isolated_execute(), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    train.main(["--arm", arm, "--fold", "0", "--seed", "0",
                                "--output", "outputs/preflight/guard_test", "--execute"])

    def test_execute_rejects_output_outside_v8(self):
        with tempfile.TemporaryDirectory() as directory, self._isolated_execute(), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "v8/outputs"):
                train.main(["--arm", "joint", "--output", directory, "--execute"])

    def test_scratch_arms_reject_source_checkpoint(self):
        with self._isolated_execute(), redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                train.main(["--arm", "joint", "--fold", "0", "--seed", "0",
                            "--output", "outputs/preflight/guard_test",
                            "--source-checkpoint", "whatever.pt"])


class SyntheticDependencyBudgetTests(unittest.TestCase):
    """完整合成缓存/池/源清单下：9 个深度臂预算照旧，非深度臂不冒充梯度训练。"""

    @staticmethod
    def _touch(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def _prepare_root(self, root):
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (config_dir / "stage3.json").write_text(
            (Path(train.ROOT) / "configs/stage3.json").read_text(encoding="utf-8"),
            encoding="utf-8")

    def _synthetic_dependencies(self):
        units = [f"{bearing}/{condition}" for bearing in PRIMARY_BEARINGS
                 for condition in CONDITIONS]
        self.assertEqual(len(units), 116)
        pools = {"all": list(units), "K19_old": units[:19], "K19_gg": units[19:38],
                 "K116": list(units)}
        recordings = [SimpleNamespace(unit_key=f"12k_DE/f{index:03d}.mat")
                      for index in range(153) for _ in range(2)]
        source_records = [{"unit_key": recording.unit_key,
                           "channel": "DE" if index % 2 == 0 else "FE",
                           "start": index * 10, "stop": index * 10 + 10}
                          for index, recording in enumerate(recordings)]
        target_records = [{"unit_key": unit, "channel": "vibration_1",
                           "start": index * 20, "stop": index * 20 + 20}
                          for index, unit in enumerate(pools["K19_old"])]
        source_units = sorted({recording.unit_key for recording in recordings})
        splits = [{"train": source_units[:115], "val": source_units[115:]}
                  for _ in range(4)]

        def load_cache(dataset, cache_dir):
            return SimpleNamespace(records=list(source_records if dataset == "cwru"
                                                else target_records))

        return pools, load_cache, recordings, splits

    @contextmanager
    def _synthetic_environment(self, root, pools, load_cache, recordings, splits):
        """显式合成依赖：覆盖缺依赖 mock，验证有依赖时的预算字段且禁止真实训练。"""
        with mock.patch.object(train, "ROOT", root), \
                mock.patch.object(train, "_load_pools", return_value=pools), \
                mock.patch.object(train, "load_cache", side_effect=load_cache), \
                mock.patch("datasets.cwru.enumerate_recordings", return_value=recordings), \
                mock.patch("datasets.cwru.split_source_units", return_value=splits), \
                forbidden_training():
            yield

    def _plan(self, arm, environment):
        root, pools, load_cache, recordings, splits = environment
        with self._synthetic_environment(root, pools, load_cache, recordings, splits):
            args = train.build_parser().parse_args(["--arm", arm, "--fold", "0", "--seed", "0"])
            _, _, plan, _, _ = train.prepare_plan(args)
        return plan

    def test_execute_guard_forbids_real_training_with_synthetic_dependencies(self):
        """合成依赖齐备时 --execute 也必须停在模型构造前，绝不真实拟合。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_root(root)
            pools, load_cache, recordings, splits = self._synthetic_dependencies()
            output = root / "outputs" / "train" / "joint_f0_s0"
            with self._synthetic_environment(root, pools, load_cache, recordings, splits):
                with self.assertRaisesRegex(AssertionError, "禁止构造模型"):
                    train.main(["--arm", "joint", "--fold", "0", "--seed", "0",
                                "--execute", "--output", str(output)])

    def test_deep_arm_budgets_stay_unchanged(self):
        epochs = json.loads((Path(train.ROOT) / "configs/stage3.json").read_text(
            encoding="utf-8"))["parameters"]["epochs"]
        expected = {
            "joint": (0, None, "none", "source_only"),
            "joint_logspec": (0, None, "none", "source_only"),
            "mixup4src": (0, None, "none", "source_only"),
            "coral4src": (0, None, "none", "source_only"),
            "adabn": (0, None, "uniform", "transductive"),
            "ce_K19": (19, "K19_old", "balanced", "supervised_contains_training_pool"),
            "ce_K19_gg": (19, "K19_gg", "balanced", "supervised_contains_training_pool"),
            "ce_K116": (116, "K116", "balanced", "supervised_contains_training_pool"),
            "ce_K19_logspec": (19, "K19_old", "balanced", "supervised_contains_training_pool"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_root(root)
            pools, load_cache, recordings, splits = self._synthetic_dependencies()
            environment = (root, pools, load_cache, recordings, splits)
            for base in ("joint", "joint_logspec"):
                self._touch(root / f"outputs/train/{base}_f0_s0/last.pt")
            expected_steps = train.steps_per_epoch(1150, 64)
            for arm in train.DEEP_ARMS:
                with self.subTest(arm=arm):
                    plan = self._plan(arm, environment)
                    self.assertEqual(plan["status"], "ready")
                    self.assertEqual(plan["s_ref"], 1150)
                    self.assertEqual(plan["steps_per_epoch"], expected_steps)
                    self.assertEqual(plan["planned_epochs"], epochs)
                    self.assertEqual(plan["planned_steps"], expected_steps * epochs)
                    self.assertEqual(plan["num_samples_per_epoch"], expected_steps * 64)
                    self.assertEqual(plan["planned_updates"], 0 if arm == "adabn"
                                     else expected_steps * epochs)
                    usage, budget, sampling, regime = expected[arm]
                    self.assertEqual(plan["target_label_usage"], usage)
                    self.assertEqual(plan["split"]["target"]["budget"], budget)
                    self.assertEqual(plan["target_sampling"], sampling)
                    self.assertEqual(plan["evaluation_regime"], regime)
                    if arm == "adabn":
                        # 无标签 AdaBN 只用 U，不消耗目标标签预算。
                        self.assertEqual(len(plan["split"]["target"]["train"]), 116)
                    else:
                        self.assertEqual(len(plan["split"]["target"]["train"]), usage)
                    self.assertEqual(plan["source_train_unit_count"], 115)
                    self.assertEqual(plan["source_validation_unit_count"], 38)

    def test_non_deep_arms_inherit_budgets_without_gradient_plans(self):
        expected = {
            "svm_phys": (19, "K19_old", "unit_quota",
                         "supervised_contains_training_pool", "shallow.py"),
            "fusion_ce_K19": (19, "K19_old", "inherit_base",
                              "supervised_contains_training_pool", "evaluate.py"),
            "fusion_joint": (0, None, "none", "source_only", "evaluate.py"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_root(root)
            pools, load_cache, recordings, splits = self._synthetic_dependencies()
            environment = (root, pools, load_cache, recordings, splits)
            for name in train.NORMAL_FILES:
                self._touch(root / f"outputs/train/normal_f0/{name}")
            for base in ("joint", "ce_K19"):
                for dataset in ("cwru", "paderborn"):
                    for name in train.DEEP_EVAL_FILES:
                        self._touch(root / f"outputs/eval/{base}_f0_s0_{dataset}/{name}")
            for arm, (usage, budget, sampling, regime, entry) in expected.items():
                with self.subTest(arm=arm):
                    plan = self._plan(arm, environment)
                    self.assertEqual(plan["status"], "ready")
                    self.assertEqual(plan["entry"], entry)
                    self.assertEqual(plan["target_label_usage"], usage)
                    self.assertEqual(plan["split"]["target"]["budget"], budget)
                    self.assertEqual(plan["target_sampling"], sampling)
                    self.assertEqual(plan["evaluation_regime"], regime)
                    self.assertEqual(len(plan["split"]["target"]["train"]), usage)
                    self.assertIsNone(plan["planned_epochs"])
                    self.assertIsNone(plan["planned_steps"])
                    self.assertIsNone(plan["planned_updates"])
                    self.assertIsNone(plan["num_samples_per_epoch"])
                    self.assertIsNone(plan["steps_per_epoch"])
                    if usage:
                        self.assertTrue(plan["split"]["target"]["labeled"])
                    else:
                        self.assertFalse(plan["split"]["target"]["labeled"])


if __name__ == "__main__":
    unittest.main()
