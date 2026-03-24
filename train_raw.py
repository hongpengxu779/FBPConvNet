"""
train_raw.py  —— 用成对的 low/high RAW 体数据训练 FBPCONVNet
按 Z 轴切片，每张切片 (X, Y) = (652, 685) 作为一个 2D 样本。
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import FBPCONVNet


# ──────────────────── Dataset ────────────────────
class RawSliceDataset(Dataset):
    """从两个 raw 体数据中按 Z 轴抽切片，返回归一化到 [0,1] 的 (input, target) 对。"""

    def __init__(self, low_path, high_path, x, y, z, augment=True):
        arr_low = np.fromfile(low_path, dtype=np.uint16)
        arr_high = np.fromfile(high_path, dtype=np.uint16)
        expect = x * y * z
        assert arr_low.size == expect, f"low raw 体素数不匹配: {arr_low.size} vs {expect}"
        assert arr_high.size == expect, f"high raw 体素数不匹配: {arr_high.size} vs {expect}"

        # Fortran order (X fastest) → (X, Y, Z)
        vol_low = arr_low.reshape((x, y, z), order='F').astype(np.float32)
        vol_high = arr_high.reshape((x, y, z), order='F').astype(np.float32)

        # 全局归一化到 [0,1]（low 和 high 必须共用同一组 min/max，
        # 因为网络有残差连接 output = net(input) + input，
        # 需要保证两者在同一值域下残差才有意义）
        global_min = min(vol_low.min(), vol_high.min())
        global_max = max(vol_low.max(), vol_high.max())
        self.global_min = global_min
        self.global_max = global_max
        vol_low = (vol_low - global_min) / max(global_max - global_min, 1.0)
        vol_high = (vol_high - global_min) / max(global_max - global_min, 1.0)

        # 按 Z 切片存储: 每张 shape = (X, Y)
        self.slices_low = [vol_low[:, :, k] for k in range(z)]
        self.slices_high = [vol_high[:, :, k] for k in range(z)]
        self.augment = augment

        # 计算 pad 尺寸（U-Net 需要是 16 的倍数）
        self.orig_h, self.orig_w = x, y
        self.pad_h = (16 - x % 16) % 16
        self.pad_w = (16 - y % 16) % 16

        print(f"  slices: {z}, shape per slice: ({x}, {y}), "
              f"padded to ({x + self.pad_h}, {y + self.pad_w})")

    def __len__(self):
        return len(self.slices_low)

    def __getitem__(self, idx):
        inp = self.slices_low[idx].copy()   # (X, Y)
        tgt = self.slices_high[idx].copy()

        # 数据增强
        if self.augment:
            if random.random() > 0.5:
                inp = np.flip(inp, axis=0).copy()
                tgt = np.flip(tgt, axis=0).copy()
            if random.random() > 0.5:
                inp = np.flip(inp, axis=1).copy()
                tgt = np.flip(tgt, axis=1).copy()

        # pad 到 16 的整数倍
        if self.pad_h > 0 or self.pad_w > 0:
            inp = np.pad(inp, ((0, self.pad_h), (0, self.pad_w)), mode="reflect")
            tgt = np.pad(tgt, ((0, self.pad_h), (0, self.pad_w)), mode="reflect")

        # → [1, H, W]
        inp = torch.from_numpy(inp).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        return inp, tgt


# ──────────────────── Utils ──────────────────────
def save_sample_image(inp, pred, tgt, path):
    """拼接 input | prediction | target 保存为灰度 jpg"""
    imgs = []
    for t in [inp, pred, tgt]:
        a = t.squeeze().cpu().numpy()
        a = np.clip(a, 0, 1)
        a = (a * 255).astype(np.uint8)
        imgs.append(a)
    canvas = np.concatenate(imgs, axis=1)
    Image.fromarray(canvas).save(path)


def latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("raw_epoch") and f.endswith(".pkl")]
    if not files:
        return None
    files.sort(key=lambda s: int(s.split("-")[1].split(".")[0]))
    return os.path.join(ckpt_dir, files[-1])


# ──────────────────── Train ──────────────────────
def main(cfg):
    os.makedirs(cfg.sample_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 数据集
    print("loading raw volumes ...")
    dataset = RawSliceDataset(
        cfg.low_raw, cfg.high_raw,
        cfg.x, cfg.y, cfg.z,
        augment=True
    )
    # 按 9:1 划分训练/验证
    n = len(dataset)
    n_val = max(1, int(n * 0.1))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"  train: {n_train}, val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 模型
    model = FBPCONVNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_start, weight_decay=1e-8)

    epoch_start = 0
    if cfg.resume:
        ckpt_path = latest_checkpoint(cfg.checkpoint_dir)
        if ckpt_path is not None:
            print("resume from:", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print("skip optimizer state:", e)
            epoch_start = ckpt.get("epoch", 0)

    # 学习率衰减
    lr_schedule = np.logspace(np.log10(cfg.lr_start), np.log10(cfg.lr_end), cfg.epoch)

    print(f"start training from epoch {epoch_start} ...")
    best_val_loss = float("inf")

    for e in range(epoch_start, cfg.epoch):
        lr = float(lr_schedule[min(e, len(lr_schedule) - 1)])
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for i, (inp, tgt) in enumerate(train_loader, start=1):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max)
            optimizer.step()

            train_loss_sum += loss.item() * inp.size(0)
            train_count += inp.size(0)

            if i % cfg.log_step == 0:
                print(f"  [epoch {e+1}/{cfg.epoch}] iter {i}/{len(train_loader)} "
                      f"loss={loss.item():.7f} lr={lr:.6f}")

            # 保存样本图
            if i % cfg.sample_step == 0:
                with torch.no_grad():
                    sp = os.path.join(cfg.sample_dir, f"raw_e{e+1}_i{i}.jpg")
                    save_sample_image(inp[0], pred[0], tgt[0], sp)

        avg_train = train_loss_sum / max(train_count, 1)

        # ---- validate ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                pred = model(inp)
                loss = criterion(pred, tgt)
                val_loss_sum += loss.item() * inp.size(0)
                val_count += inp.size(0)
        avg_val = val_loss_sum / max(val_count, 1)

        print(f"epoch {e+1}/{cfg.epoch}  train_loss={avg_train:.7f}  "
              f"val_loss={avg_val:.7f}  lr={lr:.6f}")

        # ---- checkpoint ----
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val

        if (e + 1) % cfg.ckpt_step == 0 or (e + 1) == cfg.epoch or is_best:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"raw_epoch-{e+1}.pkl")
            torch.save({
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": avg_val,
            }, ckpt_path)
            tag = " [best]" if is_best else ""
            print(f"  saved: {ckpt_path}{tag}")

    print("training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FBPCONVNet on paired raw volumes")

    # 数据
    parser.add_argument("--low_raw", type=str,
                        default=r"E:\xu\CT\DataSets\321_357_1400_0.14185_low.raw")
    parser.add_argument("--high_raw", type=str,
                        default=r"E:\xu\CT\DataSets\321_357_1400_0.14185_high.raw")
    parser.add_argument("--x", type=int, default=321)
    parser.add_argument("--y", type=int, default=357)
    parser.add_argument("--z", type=int, default=1400)

    # 训练
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="每张 652x688(padded) 约占 ~1.2 MB 显存，按GPU酌情调大")
    parser.add_argument("--lr_start", type=float, default=1e-3)
    parser.add_argument("--lr_end", type=float, default=1e-5)
    parser.add_argument("--grad_max", type=float, default=0.1)

    # 日志 & 保存
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--sample_step", type=int, default=50)
    parser.add_argument("--sample_dir", type=str, default="./samples_raw/")
    parser.add_argument("--ckpt_step", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--resume", action="store_true", default=True)

    cfg = parser.parse_args()
    print(cfg)
    main(cfg)
