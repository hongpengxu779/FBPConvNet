# FBPConvNet 精度优化策略

> 当前问题：模型可以去除稀疏角度重建的条纹伪影，但输出图像的细节（边缘锐度、微小结构）
> 相比高质量 GT 仍有明显差距，整体偏模糊/平滑。

---

## 一、根因分析

| 问题 | 现有实现 | 影响 |
|------|---------|------|
| **损失函数** | 仅 MSELoss | MSE 倾向于回归均值 → 输出过度平滑，高频细节被抑制 |
| **下采样** | MaxPool2d | 不可学习，丢弃 75% 空间信息，细节永久丢失 |
| **上采样** | ConvTranspose2d | 容易产生棋盘格伪影 |
| **激活函数** | ReLU | 存在"死神经元"问题，负值信息被截断 |
| **注意力机制** | 无 | 网络无法自适应地关注重要区域和通道 |
| **数据增强** | 仅水平/垂直翻转 | 多样性不足，模型泛化能力受限 |
| **学习率策略** | log-space 固定衰减 | 无法在训练后期精细搜索最优解 |

---

## 二、优化策略及对应代码修改

### 策略 1：混合损失函数（最高优先级）⭐

**原理**：纯 MSE 是逐像素度量，对结构/边缘不敏感。引入：
- **L1 Loss**：比 MSE 对异常值更鲁棒，保留更多纹理
- **SSIM Loss**：结构相似性，从亮度/对比度/结构三维度衡量图像质量
- **Edge Loss（Sobel 梯度）**：显式约束边缘和高频细节的一致性

**公式**：
$$\mathcal{L} = \lambda_1 \cdot L_1 + \lambda_2 \cdot (1 - \text{SSIM}) + \lambda_3 \cdot L_{\text{edge}}$$

推荐初始权重：$\lambda_1 = 1.0,\ \lambda_2 = 0.5,\ \lambda_3 = 0.1$

**代码**：新建 `losses.py`，包含 `SSIMLoss`、`EdgeLoss`、`CombinedLoss` 三个类。
在 `train_raw.py` 中将 `criterion = nn.MSELoss()` 替换为 `CombinedLoss`。

---

### 策略 2：改进模型架构（高优先级）⭐

**改动点 a — 用可学习的 Strided Convolution 替换 MaxPool**

MaxPool 丢弃信息。换成 stride=2 的卷积，让网络自己学习如何下采样：
```python
# 之前：nn.MaxPool2d(kernel_size=2)
# 之后：nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
```

**改动点 b — 用 Upsample + Conv 替换 ConvTranspose2d**

避免棋盘格伪影：
```python
# 之前：ConvTranspose2d(in_ch, out_ch, k=3, stride=2, padding=1, output_padding=1)
# 之后：
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
```

**改动点 c — 添加通道注意力 (Channel Attention)**

在每个编码器/解码器 block 末尾加入 SE (Squeeze-and-Excitation) 模块：
```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w
```

**改动点 d — 用 LeakyReLU 替换 ReLU**

保留负值信息：`nn.LeakyReLU(0.1)` 替换所有 `nn.ReLU()`。

**代码**：新建 `model_v2.py`，包含以上改进的 `FBPCONVNetV2`。

---

### 策略 3：增强数据增强

在原有翻转基础上添加：
- **随机旋转 (90°/180°/270°)**：CT 切片旋转不变
- **随机裁剪 + Resize**：强迫网络学习多尺度特征

```python
# 随机 90° 旋转
k = random.randint(0, 3)
inp = np.rot90(inp, k).copy()
tgt = np.rot90(tgt, k).copy()
```

---

### 策略 4：学习率策略

用 **Cosine Annealing with Warm Restart** 替换 log-space 衰减，训练后期周期性回升有利于跳出局部最优：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
```

---

### 策略 5：梯度裁剪 & 优化器

- 将 `grad_max` 从 0.1 放宽到 **1.0**（0.1 过于激进，限制了大梯度更新）
- 可尝试 **AdamW**（解耦权重衰减，泛化更好）

---

## 三、优先级路线图

| 阶段 | 改动 | 预期提升 | 风险 |
|------|------|---------|------|
| **Phase 1** | 混合损失 (L1+SSIM+Edge) | PSNR +1~3 dB，细节明显改善 | 低 |
| **Phase 2** | 模型架构改进 (V2) | 边缘更锐利，减少伪影 | 中 —需重新训练 |
| **Phase 3** | 数据增强 + LR 策略 | 泛化性提升，收敛更稳定 | 低 |
| **Phase 4** | 对抗训练 (可选) | 纹理更真实（但可能引入假细节） | 高 |

---

## 四、实际代码修改清单

以下文件已被修改/新增：

| 文件 | 状态 | 说明 |
|------|------|------|
| `losses.py` | **新增** | SSIMLoss、EdgeLoss、CombinedLoss |
| `model_v2.py` | **新增** | 改进架构：注意力 + stridedConv + LeakyReLU |
| `train_raw.py` | **修改** | 使用 CombinedLoss、支持选择 V1/V2 模型、增强数据增强、cosine LR |
| `infer_raw.py` | **修改** | 支持加载 V2 模型 |

> 原有 `model.py` 保持不变，可随时切回原始版本对比。
