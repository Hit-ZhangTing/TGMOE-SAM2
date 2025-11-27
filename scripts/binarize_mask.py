import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def binarize_all_masks(root_dir):
    """
    递归遍历 root_dir，将所有 PNG mask 转换成 0/1 灰度图
    """
    png_files = []

    # 遍历所有 PNG 文件
    for base, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".png"):
                png_files.append(os.path.join(base, f))

    print(f"发现 PNG 文件数量: {len(png_files)}")

    for path in tqdm(png_files):
        mask = np.array(Image.open(path))

        # 如果是三通道（RGB），取第一通道
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # 二值化：非0全部变为1
        mask_bin = (mask > 0).astype(np.uint8)

        # 保存灰度图
        Image.fromarray(mask_bin, mode="L").save(path)

    print("完成！所有 mask 已转换为 0/1.")

if __name__ == "__main__":
    # 修改为你的 DAVIS 输出路径
    root = "/home/y530/zt/SAMWISE/data/ref-youtube-vos/train/Annotations"

    binarize_all_masks(root)
