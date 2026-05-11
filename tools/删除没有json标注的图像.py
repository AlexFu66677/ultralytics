import os
from tqdm import tqdm

# ========= 配置 =========
DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\labelme"  # 图像和json在同一目录

# 支持的图像格式
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ========= 主流程 =========
files = os.listdir(DIR)

for file in tqdm(files):
    if not file.lower().endswith(IMG_EXTS):
        continue

    img_path = os.path.join(DIR, file)

    base = os.path.splitext(file)[0]
    json_path = os.path.join(DIR, base + ".json")

    # 👉 没有对应json → 删除图像
    if not os.path.exists(json_path):
        os.remove(img_path)
        print(f"🗑 删除: {file}")

print("Done!")