# -*- coding: utf-8 -*-
"""Paderborn / KAt 32 个轴承的事实表（官方事实表逐项转录，已核）。

事实来源：每个 `.rar` 内自带的 `<bearing>.pdf`（官方事实表）与 `measuring_log_<bearing>.pdf`，
以及 Lessmeier et al. (2016), PHM Society European Conference 3(1) 的 Table 4 / 5 / 6 / 7。
详见 `docs/数据事实_CWRU与Paderborn.md` §2.1。

字段含义：
- `fault_type`：官方 `K`=健康、`KA`=外圈（Außenring）、`KI`=内圈（Innenring）、`KB`=内外圈复合
- `origin`：`artificial`（EDM／钻孔／电刻笔）或 `real`（加速寿命试验产生的 pitting／indentation）
- `severity`：官方损伤程度等级 1–3，健康为 0（PHME16 Table 2）
- `damage`：损伤方式
- `pitch_diameter_mm`：**逐轴承不同**（FAG 28.55，IBU/MTK 29.05），算 BPFO/BPFI 必须用它

**类别映射到 v6 的 4 分类空间**（`LABEL_MAP = {N:0, B:1, IR:2, OR:3}`）：
- `normal` → `N`（0）
- `inner` → `IR`（2）
- `outer` → `OR`（3）
- `inner+outer`（KB23/24/27）→ **映射为 `None`**：官方按"主导损伤"人为指派（KB23/24→内圈、KB27→外圈），
  不是真值。主口径排除这 3 个轴承（决策 D4），只在事后分析里报告模型把它们判成了什么。
"""
from __future__ import annotations

from dataclasses import dataclass

#: 轴承几何（6203）：8 个球、球径 6.75 mm、接触角 0
BALL_COUNT = 8
BALL_DIAMETER_MM = 6.75

#: 4 分类标签顺序，必须与 v6 `dataset.LABEL_NAMES` 一致。
LABEL_NAMES = ("Normal", "Ball", "InnerRace", "OuterRace")
LABEL_MAP = {"N": 0, "B": 1, "IR": 2, "OR": 3}


@dataclass(frozen=True)
class Bearing:
    bearing_id: str
    manufacturer: str
    fault_type: str          # normal | outer | inner | inner+outer
    origin: str              # artificial | real | none
    severity: int            # 0（健康）| 1 | 2 | 3
    damage: str
    pitch_diameter_mm: float

    @property
    def label_letter(self) -> str | None:
        """映射到 v6 的类别字母；KB 复合故障返回 None（见模块文档）。"""
        return {"normal": "N", "inner": "IR", "outer": "OR"}.get(self.fault_type)

    @property
    def label(self) -> int | None:
        letter = self.label_letter
        return None if letter is None else LABEL_MAP[letter]

    @property
    def is_kb(self) -> bool:
        return self.fault_type == "inner+outer"

    @property
    def fault_orders(self) -> tuple[float, float]:
        """返回 (BPFO, BPFI) 的转频倍数。6203 接触角 0。"""
        ratio = BALL_DIAMETER_MM / self.pitch_diameter_mm
        half = BALL_COUNT / 2
        return half * (1 - ratio), half * (1 + ratio)


def _b(bearing_id, manufacturer, fault_type, origin, severity, damage):
    pitch = 28.55 if manufacturer == "FAG" else 29.05
    return Bearing(bearing_id, manufacturer, fault_type, origin, severity, damage, pitch)


# ---------------------------------------------------------------------------
# CWRU 来源几何（计划 §2.1 冻结）：故障来源轴承与观测测点分开。
# 来源位置是已知试验元数据，不是诊断类别；禁止按标签/严重度/损伤来源改公式。
# 数值是转频倍数，使用时乘 RPM/60。DE 沿用 v1 常数。
# ---------------------------------------------------------------------------
CWRU_DE_ORDERS = {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 2.3572, "FTF": 0.3983}


def _cwru_fe_orders():
    """FE 按 8 球、接触角 0，由已记录的 BPFO=3.0530 派生（计划 §2.1）。

    r=1-2*BPFO/Z=0.23675；BPFI=Z*(1+r)/2、BSF=(1-r*r)/(2*r)、FTF=(1-r)/2。
    历史 3.9874 约对应 2BSF，不再当 BSF 混用。这不是新增官网核验。
    """
    balls = 8
    bpfo = 3.0530
    ratio = 1 - 2 * bpfo / balls
    return {"BPFO": bpfo, "BPFI": balls * (1 + ratio) / 2,
            "BSF": (1 - ratio * ratio) / (2 * ratio), "FTF": (1 - ratio) / 2}


CWRU_FE_ORDERS = _cwru_fe_orders()


#: 32 个轴承，按官方文件索引顺序。健康 K001–K006 的 run-in 时长/载荷见 PHME16 Table 7。
BEARINGS: tuple[Bearing, ...] = (
    _b("K001", "IBU", "normal", "none", 0, "none"),
    _b("K002", "IBU", "normal", "none", 0, "none"),
    _b("K003", "IBU", "normal", "none", 0, "none"),
    _b("K004", "IBU", "normal", "none", 0, "none"),
    _b("K005", "IBU", "normal", "none", 0, "none"),
    _b("K006", "IBU", "normal", "none", 0, "none"),
    _b("KA01", "MTK", "outer", "artificial", 1, "EDM"),
    _b("KA03", "MTK", "outer", "artificial", 2, "electric engraver"),
    _b("KA05", "IBU", "outer", "artificial", 1, "electric engraver"),
    _b("KA06", "IBU", "outer", "artificial", 2, "electric engraver"),
    _b("KA07", "IBU", "outer", "artificial", 1, "drilling"),
    _b("KA08", "IBU", "outer", "artificial", 2, "drilling"),
    _b("KA09", "IBU", "outer", "artificial", 2, "drilling"),
    _b("KA04", "FAG", "outer", "real", 1, "pitting"),
    _b("KA15", "FAG", "outer", "real", 1, "indentation"),
    _b("KA16", "MTK", "outer", "real", 2, "pitting"),
    _b("KA22", "IBU", "outer", "real", 1, "pitting"),
    _b("KA30", "MTK", "outer", "real", 1, "indentation"),
    _b("KI01", "MTK", "inner", "artificial", 1, "EDM"),
    _b("KI03", "MTK", "inner", "artificial", 1, "electric engraver"),
    _b("KI05", "IBU", "inner", "artificial", 1, "electric engraver"),
    _b("KI07", "IBU", "inner", "artificial", 2, "electric engraver"),
    _b("KI08", "IBU", "inner", "artificial", 2, "electric engraver"),
    _b("KI04", "MTK", "inner", "real", 1, "pitting"),
    _b("KI14", "MTK", "inner", "real", 1, "pitting"),
    _b("KI16", "FAG", "inner", "real", 3, "pitting"),
    _b("KI17", "MTK", "inner", "real", 1, "pitting"),
    _b("KI18", "MTK", "inner", "real", 2, "pitting"),
    _b("KI21", "FAG", "inner", "real", 1, "pitting"),
    _b("KB23", "MTK", "inner+outer", "real", 2, "pitting"),
    _b("KB24", "FAG", "inner+outer", "real", 3, "pitting"),
    _b("KB27", "FAG", "inner+outer", "real", 1, "indentation"),
)

BY_ID = {bearing.bearing_id: bearing for bearing in BEARINGS}

#: 四个工况：名称 -> (转频 Hz, 转速 rpm, 负载扭矩 Nm, 径向力 N)。PHME16 Table 6。
CONDITIONS = {
    "N15_M07_F10": (25.0, 1500, 0.7, 1000),
    "N09_M07_F10": (15.0, 900, 0.7, 1000),
    "N15_M01_F10": (25.0, 1500, 0.1, 1000),
    "N15_M07_F04": (25.0, 1500, 0.7, 400),
}

#: 官方 KA08.rar 里被截断并补零的文件（MATLAB 打不开，vibration 只剩前 2.98 s）。
TRUNCATED_RECORDINGS = {"N15_M01_F10_KA08_2"}

#: 主口径（决策 D4）：排除 3 个 KB 复合故障轴承，剩 29 个。
PRIMARY_BEARINGS = tuple(b.bearing_id for b in BEARINGS if not b.is_kb)


def _self_check() -> None:
    """锁死事实表的不变量：计数、类别映射、逐厂商节径。"""
    from collections import Counter

    assert len(BEARINGS) == 32, len(BEARINGS)
    assert len(BY_ID) == 32, "轴承号重复"
    assert len(PRIMARY_BEARINGS) == 29, len(PRIMARY_BEARINGS)

    by_type = Counter(b.fault_type for b in BEARINGS)
    assert by_type == {"normal": 6, "outer": 12, "inner": 11, "inner+outer": 3}, by_type

    by_origin = Counter(b.origin for b in BEARINGS)
    assert by_origin == {"none": 6, "artificial": 12, "real": 14}, by_origin

    artificial = {b.fault_type for b in BEARINGS if b.origin == "artificial"}
    assert artificial == {"outer", "inner"}, "人工损伤只在外圈/内圈"
    assert not any(b.fault_type == "inner+outer" and b.origin == "artificial" for b in BEARINGS)

    # 厂商与节径必须一致，且只有 FAG 用 28.55
    for bearing in BEARINGS:
        expected = 28.55 if bearing.manufacturer == "FAG" else 29.05
        assert bearing.pitch_diameter_mm == expected, bearing.bearing_id

    # 类别映射：主口径三类计数
    primary = [BY_ID[k] for k in PRIMARY_BEARINGS]
    labels = Counter(b.label_letter for b in primary)
    assert labels == {"N": 6, "IR": 11, "OR": 12}, labels

    # KB 不参与映射
    assert all(b.label is None for b in BEARINGS if b.is_kb)

    # 阶次：6203 应落在 BPFO≈3.06、BPFI≈4.94
    for bearing in BEARINGS:
        bpfo, bpfi = bearing.fault_orders
        assert 3.05 < bpfo < 3.08 and 4.92 < bpfi < 4.95, (bearing.bearing_id, bpfo, bpfi)

    # CWRU 来源几何：DE 冻结常数；FE 由 BPFO 派生且 BSF≈1.993557418（不是 2BSF≈3.9874）
    assert CWRU_DE_ORDERS == {"BPFO": 3.5848, "BPFI": 5.4152, "BSF": 2.3572, "FTF": 0.3983}
    assert abs(CWRU_FE_ORDERS["BPFO"] - 3.0530) < 1e-12
    assert abs(CWRU_FE_ORDERS["BPFI"] - 4.9470) < 1e-12
    assert abs(CWRU_FE_ORDERS["BSF"] - 1.993557418) < 1e-6
    assert abs(CWRU_FE_ORDERS["FTF"] - 0.381625) < 1e-6

    print("轴承事实表自检通过")
    print(f"  主口径 29 个：{dict(labels)}")
    print(f"  排除的 KB：{[b.bearing_id for b in BEARINGS if b.is_kb]}")
    print(f"  CWRU DE 阶次：{CWRU_DE_ORDERS}")
    print(f"  CWRU FE 阶次：{CWRU_FE_ORDERS}")


if __name__ == "__main__":
    _self_check()
