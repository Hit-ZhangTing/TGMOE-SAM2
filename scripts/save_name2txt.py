import os
from pathlib import Path

def save_subfolder_names_to_txt(root_directory: str, output_filename: str = "folder_names.txt"):
    """
    遍历指定根目录下的所有一级子文件夹，并将它们的名称保存到指定的 TXT 文件中。

    Args:
        root_directory (str): 包含子文件夹的根目录路径。
        output_filename (str): 输出 TXT 文件的文件名。
    """
    root_path = Path(root_directory)
    output_path = root_path / output_filename

    if not root_path.is_dir():
        print(f"❌ 错误: 路径 '{root_directory}' 不是一个有效的目录。")
        return

    print(f"--- 正在扫描目录: {root_directory} ---")

    subfolder_names = []
    
    # 遍历根目录下的所有内容
    for item in root_path.iterdir():
        # 检查是否是目录（子文件夹）
        if item.is_dir():
            subfolder_names.append(item.name)

    if not subfolder_names:
        print("🔍 目录中未找到任何子文件夹。")
        return

    # 对子文件夹名称进行排序（可选，但推荐保持输出一致性）
    subfolder_names.sort()

    # 将名称写入 TXT 文件，每个名称占一行
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(subfolder_names))
            
        print(f"✅ 成功写入 {len(subfolder_names)} 个文件夹名称到: {output_path}")

    except IOError as e:
        print(f"❌ 写入文件时发生错误: {e}")

# --- 使用示例 ---
if __name__ == '__main__':
    # -----------------------------------------------------------
    # !!! 请将此路径替换为您要处理的根文件夹的实际路径 !!!
    # -----------------------------------------------------------
    target_directory = "/home/y530/zt/TGMOESAM2/data/CVC/valid/JPEGImages"
    
    # 输出文件名将生成在 target_directory 内部
    output_file = "/home/y530/zt/TGMOESAM2/data/CVC_davis/DAVIS/ImageSets/2017/val.txt"

    save_subfolder_names_to_txt(
        root_directory=target_directory,
        output_filename=output_file
    )