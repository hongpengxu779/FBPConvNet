"""
losses.py  —— 面向 CT 重建的混合损失函数
包含：SSIMLoss、EdgeLoss（Sobel 梯度）、CombinedLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────
#  SSIM Loss
# ────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """
    1 - SSIM（越小越好）。
    window_size: 高斯窗口大小（奇数），推荐 7 或 11。
    """

    def __init__(self, window_size: int = 7, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        # 创建 1-D 高斯核 → 2-D
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss_1d /= gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)  # outer product
        # shape: [1, 1, window_size, window_size]
        self.register_buffer("window", gauss_2d.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(pred.device)
        channels = pred.size(1)
        if window.size(1) != channels:
            window = window.expand(-1, channels, -1, -1)

        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1.0 - ssim_map.mean()


# ────────────────────────────────────────────────
#  Edge Loss（Sobel 梯度差异）
# ────────────────────────────────────────────────
class EdgeLoss(nn.Module):
    """
    用固定 Sobel 算子提取梯度，计算 pred 与 target 梯度的 L1 差异。
    显式约束边缘 / 高频信息一致性。
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, img: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(img, self.sobel_x.to(img.device), padding=1)
        gy = F.conv2d(img, self.sobel_y.to(img.device), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient(pred), self._gradient(target))


# ────────────────────────────────────────────────
#  Combined Loss
# ────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    L = λ1 * L1 + λ2 * (1-SSIM) + λ3 * EdgeLoss

    推荐默认值: λ1=1.0, λ2=0.5, λ3=0.1
    """

    def __init__(self, lambda_l1: float = 1.0, lambda_ssim: float = 0.5,
                 lambda_edge: float = 0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=7)
        self.edge = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_edge = self.edge(pred, target)
        return self.lambda_l1 * loss_l1 + self.lambda_ssim * loss_ssim + self.lambda_edge * loss_edge
