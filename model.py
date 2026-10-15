# -*- coding: utf-8 -*-
"""v8 自包含 MSRS-ABiLSTM（由 v6 `model.py` 逐字段复制，仅放开输入通道 1→2）。

v8 阶段三固定双通道输入 (B,2,256)：第一通道 logspec256，第二通道前 25 维缩放后
物理摘要、后 231 位补零（组装见 `features.dual_input`）。网络结构、四类头、
256 维嵌入与旧权重命名保持不变；`n_channels=1` 仍可用于输入消融自检。

数据流（自上而下）：
    输入 (B,2,256)   双通道窗口特征，一个样本即一个 512 点窗口
      │
    ① ConvFrontEnd —— 3 级 SMCBlock(32,64,128) + MaxPool(2)→MaxPool(4)→AdaptiveMaxPool1d(8)
    ② ResidualShrinkageUnit × 2   通道软阈值去噪，形状保持 (B,128,8)
    ③ SE 通道注意力（use_se=True）形状不变
    ④ BiLSTM（单层双向，hidden=128） (B,128,8)→(B,8,256)
    ⑤ AdditiveAttention → 上下文 (B,256) + 权重 (B,8)
    ⑥ 分类头 FC(256→64)→ReLU→Dropout(0.5)→FC(64→4)

v6 的 MMD/监督对比损失函数按原样保留在文件尾部（阶段三各臂不调用，
仅供历史对照），阶段三新增的 Mixup/CORAL 损失在 `transfer.py`。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SMCBlock(nn.Module):
    """并联多尺度卷积（k=3/7/15）+ 1×1 融合。输入 (B,C_in,L)→(B,C_out,L)，same padding。"""

    def __init__(self, c_in, c_out, kernels=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2,
                      groups=c_in if c_in == c_out else 1)
            for kernel in kernels
        ])
        self.pointwise = nn.Conv1d(c_out * len(kernels), c_out, 1)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        values = torch.cat([branch(x) for branch in self.branches], dim=1)
        return F.relu(self.bn(self.pointwise(values)))


class ConvFrontEnd(nn.Module):
    """谱前端：3 级 SMCBlock(32,64,128) + 池化，把 (B,1,256) 压到 (B,128,8)。

    首级必须 /2 而不是 /4，为保住低频故障特征带的可分性；末级自适应只做长度归一。
    """

    def __init__(self, c_in=1, channels=(32, 64, 128)):
        super().__init__()
        blocks, pools = [], []
        previous = c_in
        for index, current in enumerate(channels):
            blocks.append(SMCBlock(previous, current))
            pools.append(nn.MaxPool1d(2 if index == 0 else 4) if index < 2 else nn.AdaptiveMaxPool1d(8))
            previous = current
        self.blocks = nn.ModuleList(blocks)
        self.pools = nn.ModuleList(pools)
        self.out_channels = previous

    def forward(self, x):
        for block, pool in zip(self.blocks, self.pools):
            x = pool(block(x))
        return x


class ResidualShrinkageUnit(nn.Module):
    """残差块 + 通道自适应软阈值去噪，输入输出同形 (B,C,L)。"""

    def __init__(self, channels, hidden=16):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.threshold = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(),
            nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        scale = self.threshold(residual.mean(dim=2)).unsqueeze(-1)
        residual = torch.sign(residual) * F.relu(residual.abs() - scale)
        return F.relu(x + residual)


class AdditiveAttention(nn.Module):
    """Bahdanau 加性注意力：输入 (B,8,D)→上下文 (B,D) + 权重 (B,8)。"""

    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, hidden):
        scores = self.score(torch.tanh(self.projection(hidden))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


class MSRSABiLSTM(nn.Module):
    """MSRS-ABiLSTM：输入 (B,n_channels,256)，输出四分类 logits、注意力和 256 维特征。

    v8 固定四类；输入通道只允许 2（正式双通道）或 1（输入消融自检），其余结构不动。
    """

    def __init__(self, n_channels=2, n_classes=4, lstm_hidden=128, use_se=True):
        super().__init__()
        if n_channels not in (1, 2) or n_classes != 4:
            raise ValueError("v8 固定四类，输入通道只允许 2（正式）或 1（消融自检）")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.frontend = ConvFrontEnd(n_channels)
        channels = self.frontend.out_channels
        self.rsu = nn.Sequential(
            ResidualShrinkageUnit(channels),
            ResidualShrinkageUnit(channels),
        )
        self.use_se = bool(use_se)
        if self.use_se:
            self.se = nn.Sequential(
                nn.Linear(channels, channels // 4), nn.ReLU(),
                nn.Linear(channels // 4, channels), nn.Sigmoid()
            )
        self.lstm = nn.LSTM(channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(lstm_hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, return_feat=False):
        values = self.frontend(x)
        values = self.rsu(values)
        if self.use_se:
            weights = self.se(values.mean(dim=2)).unsqueeze(-1)
            values = values * weights
        sequence, _ = self.lstm(values.permute(0, 2, 1))
        feature, attention = self.attention(sequence)
        logits = self.head(feature)
        return (logits, attention, feature) if return_feat else (logits, attention)


# ---------------------------------------------------------------------------
# 迁移损失：固定三带宽 RBF 的多核 MMD，只做逐类对齐
# ---------------------------------------------------------------------------
def gaussian_kernel(x, y, sigma=(1, 5, 10)):
    """固定三带宽多核 RBF：输入 (m,d)/(n,d)，返回 (m,n) 核矩阵。

    sigma=(1,5,10) 为冻结配方（沿用 v4/v5），三核等权平均，勿改。
    平方距离用 xx+yy-2xy 展开算，与 v5 固定核路径逐位一致（不做负误差截零，
    截零只属于已删除的 median 分支）。
    """
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    distances = xx + yy - 2 * x @ y.transpose(0, 1)
    return sum(torch.exp(-distances / (2 * value)) for value in sigma) / len(sigma)


def mmd_loss(source, target, estimator="biased", sigma=(1, 5, 10)):
    """MMD²=E[k(s,s')]+E[k(t,t')]-2E[k(s,t)]，默认 biased（有限 batch 非负）。

    任一域为空直接回设备零；unbiased 要求两域各≥2 样本，否则回退 biased。结果 clamp 非负。
    """
    if source.size(0) < 1 or target.size(0) < 1:
        return source.new_zeros(())
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"未知 MMD 估计器：{estimator}")
    if estimator == "unbiased" and (source.size(0) < 2 or target.size(0) < 2):
        estimator = "biased"
    k_ss = gaussian_kernel(source, source, sigma)
    k_tt = gaussian_kernel(target, target, sigma)
    k_st = gaussian_kernel(source, target, sigma)
    if estimator == "biased":
        return (k_ss.mean() + k_tt.mean() - 2 * k_st.mean()).clamp_min(0)
    n_s, n_t = source.size(0), target.size(0)
    ss = (k_ss.sum() - n_s) / (n_s * (n_s - 1))
    tt = (k_tt.sum() - n_t) / (n_t * (n_t - 1))
    return (ss + tt - 2 * k_st.mean()).clamp_min(0)


def classwise_mmd_loss(features_source, labels_source, features_target, labels_target,
                       shared_classes=(1, 2, 3), estimator="biased", sigma=(1, 5, 10)):
    """只对共有故障类 Ball/IR/OR 按真实标注逐类调用 mmd_loss 后等类平均。

    源 Normal（类 0）只走源 CE，不进 MMD；任一域当前类不足 2 个样本则跳过该类；
    无有效类返回源设备上的标量零，不回退全局 MMD，不加伪标签分支。
    """
    terms = []
    for label in shared_classes:
        source_mask = labels_source == label
        target_mask = labels_target == label
        if int(source_mask.sum()) < 2 or int(target_mask.sum()) < 2:
            continue
        terms.append(mmd_loss(features_source[source_mask], features_target[target_mask],
                              estimator, sigma=sigma))
    if not terms:
        return features_source.new_zeros(())
    return torch.stack(terms).mean()


# ---------------------------------------------------------------------------
# 跨域监督对比：约束类边界几何，不约束已被均衡采样器对齐的边缘分布
# ---------------------------------------------------------------------------
#: supcon 温度（冻结常用值，不进 CLI）。
SUPCON_TEMPERATURE = 0.1


def supcon_loss(features_source, labels_source, features_target, labels_target,
                temperature=SUPCON_TEMPERATURE):
    """源+目标拼接 batch 的监督对比（SupCon）：正对=同类样本，跨域同类对随之拉近。

    特征先 L2 归一化，相似度为余弦/温度；每行对全部同类对取平均 InfoNCE，
    分母含批内全部其他样本。任一域为空或批内无同类对返回设备上的标量零。
    与 classwise_mmd_loss 的区别：MMD 对齐边缘分布（均衡采样器已做），
    这里对齐条件结构（同类跨域聚拢、异类分开），正对类边界混淆。
    """
    if features_source.size(0) < 1 or features_target.size(0) < 1:
        return features_source.new_zeros(())
    features = torch.cat([features_source, features_target], dim=0)
    labels = torch.cat([labels_source, labels_target], dim=0)
    z = F.normalize(features, dim=1)
    logits = z @ z.t() / temperature
    eye = torch.eye(logits.size(0), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    if not positive.any():
        return features_source.new_zeros(())
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # 非正对位置先置零再乘 mask：对角 log_prob=-inf，0×(-inf)=NaN，直接乘会毒化整行。
    log_prob = log_prob.masked_fill(~positive, 0.0)
    per_row = -log_prob.sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return per_row.mean().clamp_min(0)  # 数学上非负，clamp 只滤 logsumexp 的 1e-8 级浮点噪声


if __name__ == "__main__":
    # 不训练自检：固定种子跑一次前向，只验形状与注意力归一。
    torch.manual_seed(0)
    model = MSRSABiLSTM(n_channels=2)
    logits, attention, feature = model(torch.randn(2, 2, 256), return_feat=True)
    assert logits.shape == (2, 4)
    assert attention.shape == (2, 8)
    assert feature.shape == (2, 256)
    assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-5)
    single = MSRSABiLSTM(n_channels=1)
    logits1, _ = single(torch.randn(2, 1, 256))
    assert logits1.shape == (2, 4)
    align = supcon_loss(torch.randn(4, 8), torch.tensor([0, 0, 1, 1]),
                        torch.randn(4, 8), torch.tensor([0, 1, 0, 1]))
    assert align.ndim == 0 and torch.isfinite(align) and align >= 0
    identical = supcon_loss(torch.zeros(4, 8), torch.tensor([0, 0, 1, 1]),
                            torch.zeros(4, 8), torch.tensor([0, 1, 0, 1]))
    # 零特征：logits 全 0，损失=均匀基线 log(n-1)=log(7)（无判别性特征被推离基线，有梯度）。
    assert abs(identical.item() - math.log(7)) < 1e-5
    print("ALL MODEL CHECKS PASSED")
