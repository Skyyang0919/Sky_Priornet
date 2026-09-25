import torch
import torch.nn as nn
import torch.nn.functional as F


class CoOccurrenceAttention(nn.Module):
    """
    可学习共存增强模块（基于类别先验的特征调制）
    通过可学习的 co_boost 矩阵，在类原型层面注入共现先验。
    """

    DEFAULT_RULES = {
    #这里是具体从数据集中统计得到的共存关系
    }

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 8,
        dropout: float = 0.1,
        co_rules=None,
        use_prior_init: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.coarse_cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1, bias=False),
        )

        # 原型投影
        self.proto_proj = nn.Linear(in_channels, in_channels)

        # 先验强度门控（可学习）
        self.alpha = nn.Parameter(torch.tensor([2.0]))

        # 共存先验矩阵：co_boost[i, j] 表示 j 出现 → i 倾向出现
        self.co_boost = nn.Parameter(torch.zeros(num_classes, num_classes))
        if use_prior_init:
            rules = co_rules if co_rules is not None else self.DEFAULT_RULES
            with torch.no_grad():
                for src, targets in rules.items():
                    if isinstance(targets, (list, tuple)):
                        for tgt in targets:
                            self.co_boost[tgt, src] = 0.6
                    elif isinstance(targets, dict):
                        for tgt, val in targets.items():
                            self.co_boost[tgt, src] = val

        # fusion：纯 PyTorch + GroupNorm
        assert in_channels % 16 == 0, (
            f"in_channels={in_channels} 
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(in_channels // 16, in_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)
        self._iter_count = 0

    def _extract_class_prototypes(self, x: torch.Tensor):
        """
        提取动态类原型并注入共存增强。

        Args:
            x: (B, C, H, W)

        Returns:
            weighted_proto: (B, num_classes, C)
            class_presence: (B, num_classes)
        """
        B, C, H, W = x.shape

        # Step 1: 软标签
        logits = self.coarse_cls_head(x)                           # (B, K, H, W)
        prob = torch.sigmoid(logits)                               # (B, K, H, W)

        # Step 2: 类原型
        prototypes = torch.einsum('bchw,bkhw->bkc', x, prob)      # (B, K, C)
        prototypes = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-6)
        prototypes = self.proto_proj(prototypes)                   # (B, K, C)
        prototypes = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-6)

        # Step 3: 各类在图中的存在度
        class_presence = prob.mean(dim=[2, 3])                     # (B, K)

        # Step 4: 先验矩阵传播共现
        boost_weights = torch.sigmoid(self.co_boost)               # (K, K)
        enhancement = torch.matmul(class_presence, boost_weights.T)  # (B, K)

        # Step 5: 用共现加权原型贡献
        alpha = F.softplus(self.alpha)
        weight = 1.0 + alpha * enhancement.unsqueeze(-1)          # (B, K, 1)
        weighted_proto = prototypes * weight                       # (B, K, C)

        return weighted_proto, class_presence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._iter_count += 1

        enhanced_proto, class_presence = self._extract_class_prototypes(x)  # (B, K, C)

        # 存在度归一化加权求和，得到场景级别的原型
        w = class_presence / (class_presence.sum(dim=1, keepdim=True) + 1e-6)
        proto_summary = torch.einsum("bk,bkc->bc", w, enhanced_proto)       # (B, C)

        # 通道注意力 + 残差融合
        channel_weight = torch.sigmoid(proto_summary).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        out = self.fusion(x * channel_weight) + x

        return out
