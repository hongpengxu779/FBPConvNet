"""
model_v2.py  —— 改进版 FBPCONVNet
相比原版 model.py 的关键改动：
  1. MaxPool → stride-2 卷积（可学习下采样，保留更多信息）
  2. ConvTranspose2d → Upsample+Conv（消除棋盘格伪影）
  3. 每个 block 末尾加入 Channel Attention (SE 模块)
  4. ReLU → LeakyReLU(0.1)（保留负值信息）
  5. 权重初始化改用 kaiming_normal_（配合 LeakyReLU）
"""
import torch
import torch.nn as nn


# ─────────────────── Channel Attention (SE) ───────────────────
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)  # 防止 mid 过小
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return x * w


# ─────────────────── Helper builders ───────────────────
def _conv_bn_act(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 bn=True, act=True):
    """Conv + (BN) + (LeakyReLU)"""
    layers = []
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
    nn.init.kaiming_normal_(conv.weight, a=0.1, nonlinearity='leaky_relu')
    nn.init.constant_(conv.bias, 0)
    layers.append(conv)
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    if act:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return layers


def _upsample_conv(in_ch, out_ch, kernel_size=3, padding=1):
    """Bilinear upsample ×2 + Conv + BN + LeakyReLU（替代 ConvTranspose2d）"""
    layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding)
    nn.init.kaiming_normal_(conv.weight, a=0.1, nonlinearity='leaky_relu')
    nn.init.constant_(conv.bias, 0)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.1, inplace=True))
    return layers


# ─────────────────── Encoder / Decoder blocks ───────────────────
class EncoderBlock(nn.Module):
    """
    (可选 stride-2 downsample) → N × Conv-BN-LeakyReLU → ChannelAttention
    """

    def __init__(self, in_ch, out_ch, n_convs=2, downsample=True):
        super().__init__()
        layers = []
        if downsample:
            # stride-2 可学习下采样（替代 MaxPool）
            layers.extend(_conv_bn_act(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            n_convs -= 1
            in_ch = out_ch
        for _ in range(n_convs):
            layers.extend(_conv_bn_act(in_ch, out_ch))
            in_ch = out_ch
        layers.append(ChannelAttention(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    N × Conv-BN-LeakyReLU → ChannelAttention → Upsample+Conv
    输入是 cat(skip, upsampled)，所以 in_ch = skip_ch + up_ch
    """

    def __init__(self, in_ch, mid_ch, out_ch, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            c_in = in_ch if i == 0 else mid_ch
            layers.extend(_conv_bn_act(c_in, mid_ch))
        layers.append(ChannelAttention(mid_ch))
        layers.extend(_upsample_conv(mid_ch, out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────── FBPCONVNetV2 ───────────────────
class FBPCONVNetV2(nn.Module):
    """
    改进版 U-Net，保持与原版相同的 4-level + bottleneck 深度。
    全局残差: output = net(input) + input
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──
        # Level 1: 1 → 64  (no downsample)
        self.enc1 = EncoderBlock(1, 64, n_convs=3, downsample=False)
        # Level 2: 64 → 128
        self.enc2 = EncoderBlock(64, 128, n_convs=2, downsample=True)
        # Level 3: 128 → 256
        self.enc3 = EncoderBlock(128, 256, n_convs=2, downsample=True)
        # Level 4: 256 → 512
        self.enc4 = EncoderBlock(256, 512, n_convs=2, downsample=True)

        # ── Bottleneck ──  512 → 1024 → upsample → 512
        bottleneck_layers = []
        bottleneck_layers.extend(_conv_bn_act(512, 1024, stride=2))
        bottleneck_layers.extend(_conv_bn_act(1024, 1024))
        bottleneck_layers.append(ChannelAttention(1024))
        bottleneck_layers.extend(_upsample_conv(1024, 512))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # ── Decoder ──
        # cat(enc4_out=512, bottleneck_up=512) = 1024 → 512, up → 256
        self.dec4 = DecoderBlock(1024, 512, 256, n_convs=2)
        # cat(enc3=256, dec4_up=256) = 512 → 256, up → 128
        self.dec3 = DecoderBlock(512, 256, 128, n_convs=2)
        # cat(enc2=128, dec3_up=128) = 256 → 128, up → 64
        self.dec2 = DecoderBlock(256, 128, 64, n_convs=2)

        # Final: cat(enc1=64, dec2_up=64) = 128 → 64 → 1  (无 upsample)
        final_layers = []
        final_layers.extend(_conv_bn_act(128, 64))
        final_layers.extend(_conv_bn_act(64, 64))
        final_layers.append(ChannelAttention(64))
        # 1×1 输出卷积，无 BN 无激活
        out_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(out_conv.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.constant_(out_conv.bias, 0)
        final_layers.append(out_conv)
        self.final = nn.Sequential(*final_layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # (B, 64,  H,   W)
        e2 = self.enc2(e1)      # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)      # (B, 256, H/4, W/4)
        e4 = self.enc4(e3)      # (B, 512, H/8, W/8)

        # Bottleneck
        bn = self.bottleneck(e4)  # (B, 512, H/8, W/8)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([e4, bn], dim=1))    # → (B, 256, H/4, W/4)
        d3 = self.dec3(torch.cat([e3, d4], dim=1))    # → (B, 128, H/2, W/2)
        d2 = self.dec2(torch.cat([e2, d3], dim=1))    # → (B, 64,  H,   W)

        out = self.final(torch.cat([e1, d2], dim=1))  # → (B, 1,   H,   W)

        # 全局残差连接
        return out + x
