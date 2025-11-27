import os
from PIL import Image

# 原始数据集路径
src_root = "/home/y530/zt/SAMWISE/data/ref-davis/DAVIS/Annotations_unsupervised/480p-bak"
# 保存的新路径
dst_root = "/home/y530/zt/SAMWISE/data/ref-davis/DAVIS/Annotations_unsupervised/480p"

MAX_IMAGES = 50  # 每个文件夹最多50张图片

# 遍历每个子文件夹
for folder_name in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # 获取该文件夹下所有 jpg 文件
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    # 按 `_` 后数字排序
    def get_sort_key(filename):
        try:
            return int(filename.split('_')[-1].split('.')[0])
        except:
            return 0

    img_files.sort(key=get_sort_key)

    # 拆分成多个子文件夹，保证每个 <=50 张
    total = len(img_files)
    num_groups = (total + MAX_IMAGES - 1) // MAX_IMAGES

    for group_idx in range(num_groups):
        # 子文件夹名称
        if group_idx == 0:
            dst_folder_name = folder_name
        else:
            dst_folder_name = f"{folder_name}_{group_idx}"

        dst_folder_path = os.path.join(dst_root, dst_folder_name)
        os.makedirs(dst_folder_path, exist_ok=True)

        # 当前组的图片列表
        start = group_idx * MAX_IMAGES
        end = min(start + MAX_IMAGES, total)
        group_imgs = img_files[start:end]

        # 重新命名并保存
        for i, img_name in enumerate(group_imgs):
            src_img_path = os.path.join(folder_path, img_name)
            img = Image.open(src_img_path)

            new_name = f"{i:05d}.png"  # 00000.jpg 开始
            dst_img_path = os.path.join(dst_folder_path, new_name)
            img.save(dst_img_path)

        print(f"文件夹 {dst_folder_name} 已完成，保存 {len(group_imgs)} 张")

print("全部处理完成！")
