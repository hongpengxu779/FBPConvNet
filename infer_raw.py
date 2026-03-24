import argparse
import numpy as np
import torch
from model import FBPCONVNet
from model_v2 import FBPCONVNetV2


def load_model(ckpt_path, device, version="v2"):
    if version == "v2":
        model = FBPCONVNetV2().to(device)
    else:
        model = FBPCONVNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_raw", type=str, default="E:\\xu\\CT\\DataSets\\321_357_1400_0.14185_low_180.raw")
    parser.add_argument("--output_raw", type=str, default="E:\\xu\\CT\\DataSets\\321_357_1400_0.14185_low_180_denoised_v2.raw")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/raw_epoch-30.pkl")
    parser.add_argument("--model_version", type=str, default="v2",
                        choices=["v1", "v2"], help="v1=原版, v2=改进版")
    parser.add_argument("--x", type=int, default=321)
    parser.add_argument("--y", type=int, default=357)
    parser.add_argument("--z", type=int, default=1400)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) 读入 raw
    arr = np.fromfile(args.input_raw, dtype=np.uint16)
    expect = args.x * args.y * args.z
    assert arr.size == expect, f"体素数不匹配: {arr.size} vs {expect}"
    vol = arr.reshape((args.x, args.y, args.z), order='F')

    # 2) 加载模型
    model = load_model(args.ckpt, device, version=args.model_version)

    # 3) 获取网络训练时的图像尺寸（用于 pad）
    #    原训练数据可能是 512x512，需要 pad 到合适尺寸
    out_vol = np.zeros_like(vol, dtype=np.float32)

    # 按 Z 轴逐切片推理（每张 X*Y = 652*685）
    for k in range(args.z):
        img = vol[:, :, k].astype(np.float32)

        # 归一化到 [0,1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img_norm = (img - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img)

        # pad 到 16 的整数倍（U-Net 需要）
        h, w = img_norm.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        img_pad = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")

        # 推理
        inp = torch.from_numpy(img_pad).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(inp).squeeze().cpu().numpy()

        # 去 pad + 反归一化
        pred = pred[:h, :w]
        pred = np.clip(pred, 0.0, 1.0)
        pred = pred * (vmax - vmin) + vmin

        out_vol[:, :, k] = pred

        if (k + 1) % 50 == 0 or (k + 1) == args.z:
            print(f"slice {k+1}/{args.z}")

    # 4) 保存（保持 Fortran order 写回）
    out_vol_u16 = np.clip(out_vol, 0, 65535).astype(np.uint16)
    out_vol_u16.flatten(order='F').tofile(args.output_raw)
    print(f"saved: {args.output_raw}")


if __name__ == "__main__":
    main()