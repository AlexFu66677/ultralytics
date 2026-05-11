import os
import shutil
from pathlib import Path

# ========= 配置 =========
src_dir = r"C:\Users\fjl\Desktop\240\raw_data\uav_combined\labelme"   # 原始文件夹
dst_dir = r"C:\Users\fjl\Desktop\240\raw_data\uav_combined"   # 输出文件夹
batch_size = 1500                  # 每个文件夹数量

# 支持的图像格式
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 创建输出目录
os.makedirs(dst_dir, exist_ok=True)

# 获取所有图像
images = [p for p in Path(src_dir).iterdir() if p.suffix.lower() in img_exts]
images.sort()  # 排序，保证顺序一致

print(f"共找到 {len(images)} 张图像")

# ========= 主逻辑 =========
for idx, img_path in enumerate(images):
    # 当前属于第几个子文件夹
    folder_idx = idx // batch_size
    subfolder = Path(dst_dir) / f"batch_{folder_idx:04d}"
    subfolder.mkdir(parents=True, exist_ok=True)

    # 目标路径
    dst_img = subfolder / img_path.name

    # 移动图像
    shutil.copy(str(img_path), str(dst_img))

    # 查找同名 json
    json_path = img_path.with_suffix(".json")
    if json_path.exists():
        dst_json = subfolder / json_path.name
        shutil.copy(str(json_path), str(dst_json))

    # 打印进度（每1000张）
    if (idx + 1) % 1000 == 0:
        print(f"已处理 {idx + 1}/{len(images)}")

print("处理完成！")