"""
check_axes.py  —— 诊断 raw 数据的 XYZ 轴定义
沿三个轴各取中间切片，保存为 PNG，人眼判断哪个方向是正确的 CT 断层。
同时测试两种内存排布：C-order (X slowest) 和 Fortran-order (X fastest)。
"""
import numpy as np
from PIL import Image
import os

# ─── 参数（按你的实际数据改）───
raw_path = r"E:\xu\CT\DataSets\323_176_285_0.2_low.raw"
X, Y, Z = 323, 176, 285
dtype = np.uint16

out_dir = "./check_axes_output"
os.makedirs(out_dir, exist_ok=True)

# 读入
arr = np.fromfile(raw_path, dtype=dtype)
expect = X * Y * Z
print(f"file voxels: {arr.size}, expect: {expect}")
assert arr.size == expect, "体素数不匹配！"

# ─── 测试两种排布 ───
orders = {
    "C_XYZ": arr.reshape((X, Y, Z), order='C'),   # X slowest, Z fastest
    "F_XYZ": arr.reshape((X, Y, Z), order='F'),   # X fastest, Z slowest
}

def norm_u8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255
    return img.astype(np.uint8)

for order_name, vol in orders.items():
    print(f"\n=== {order_name}  vol.shape = {vol.shape} ===")

    # 沿 axis=0 切（固定 X，得到 YZ 平面）
    mid0 = vol.shape[0] // 2
    s0 = vol[mid0, :, :]          # shape (Y, Z)
    Image.fromarray(norm_u8(s0)).save(os.path.join(out_dir, f"{order_name}_axis0_X{mid0}.png"))
    print(f"  axis=0 (fix X={mid0}): shape {s0.shape}  → YZ plane")

    # 沿 axis=1 切（固定 Y，得到 XZ 平面）
    mid1 = vol.shape[1] // 2
    s1 = vol[:, mid1, :]          # shape (X, Z)
    Image.fromarray(norm_u8(s1)).save(os.path.join(out_dir, f"{order_name}_axis1_Y{mid1}.png"))
    print(f"  axis=1 (fix Y={mid1}): shape {s1.shape}  → XZ plane")

    # 沿 axis=2 切（固定 Z，得到 XY 平面）
    mid2 = vol.shape[2] // 2
    s2 = vol[:, :, mid2]          # shape (X, Y)
    Image.fromarray(norm_u8(s2)).save(os.path.join(out_dir, f"{order_name}_axis2_Z{mid2}.png"))
    print(f"  axis=2 (fix Z={mid2}): shape {s2.shape}  → XY plane")

    # 额外：多切几张看看连续性
    for k in [0, vol.shape[2]//4, vol.shape[2]//2, 3*vol.shape[2]//4, vol.shape[2]-1]:
        s = vol[:, :, k]
        Image.fromarray(norm_u8(s)).save(os.path.join(out_dir, f"{order_name}_sliceZ_{k}.png"))

    for k in [0, vol.shape[0]//4, vol.shape[0]//2, 3*vol.shape[0]//4, vol.shape[0]-1]:
        s = vol[k, :, :]
        Image.fromarray(norm_u8(s)).save(os.path.join(out_dir, f"{order_name}_sliceX_{k}.png"))

print(f"\n共保存到: {os.path.abspath(out_dir)}")
print("请查看图片，找出哪组看起来是正确的 CT 断层切片。")
print("正确的 CT 断层应该：形状接近正方形/圆形，能看到物体内部结构。")
