import os
from PIL import Image
# resize目录中所有文件大小高度为480，目录中可能还有目录
def resize_images(src_dir, dst_dir, target_height=480):
    """
    递归遍历 src_dir 下所有图片文件，
    将高度缩放到 target_height，宽度等比例缩放，
    保存到 dst_dir 保持原有子目录结构
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, _, files in os.walk(src_dir):
        for file in files:
            ext = file.lower().split('.')[-1]
            if ext in {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'}:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_dir)
                dst_path = os.path.join(dst_dir, rel_path)

                # 确保目标子目录存在
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                with Image.open(src_path) as img:
                    w, h = img.size
                    new_w = int(w * target_height / h)
                    resized = img.resize((new_w, target_height), Image.LANCZOS)
                    resized.save(dst_path)
                    print(f'已处理: {src_path} -> {dst_path}')

if __name__ == '__main__':
    src_root = input('请输入源目录路径: ').strip()
    dst_root = input('请输入新目录路径: ').strip()
    resize_images(src_root, dst_root)
