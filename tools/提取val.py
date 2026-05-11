import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ===================== 路径配置 =====================
train_dir = Path(r"C:\Users\fjl\Desktop\240\train")
badcase_dir = Path(r"C:\Users\fjl\Desktop\240\badcase_raw_val")
val_dir = Path(r"C:\Users\fjl\Desktop\240\val")

# 支持的图像格式
img_exts = [".jpg", ".jpeg", ".png", ".bmp"]

# ===================== 创建目标目录 =====================
val_dir.mkdir(parents=True, exist_ok=True)

# ===================== 收集 badcase 文件名 =====================
badcase_names = set()

for f in badcase_dir.iterdir():
    if f.is_file():
        badcase_names.add(f.stem)  # 不带后缀

print(f"badcase 数量: {len(badcase_names)}")

# ===================== 开始匹配并移动 =====================
moved_count = 0

for name in tqdm(badcase_names):
    # 1️⃣ 移动图片
    for ext in img_exts:
        img_path = train_dir / f"{name}{ext}"
        if img_path.exists():
            shutil.move(str(img_path), str(val_dir / img_path.name))
            moved_count += 1
            break  # 找到一个就够了

    # 2️⃣ 移动 YOLO 标注 (.txt)
    txt_path = train_dir / f"{name}.txt"
    if txt_path.exists():
        shutil.move(str(txt_path), str(val_dir / txt_path.name))

    # 3️⃣ 移动 json 标注
    json_path = train_dir / f"{name}.json"
    if json_path.exists():
        shutil.move(str(json_path), str(val_dir / json_path.name))

print(f"移动完成，共移动图片: {moved_count}")