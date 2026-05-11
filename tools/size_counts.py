import os
from collections import Counter
from PIL import Image
from tqdm import tqdm

# ===== 配置 =====
img_dir = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\yolo\train"

# 支持的格式
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ===== 统计 =====
res_counter = Counter()
total = 0

files = []
for root, _, filenames in os.walk(img_dir):
    for f in filenames:
        if f.lower().endswith(IMG_EXTS):
            files.append(os.path.join(root, f))

for path in tqdm(files):
    try:
        with Image.open(path) as img:
            w, h = img.size
            res_counter[(w, h)] += 1
            total += 1
    except:
        print(f"读取失败: {path}")

# ===== 输出 =====
print(f"\n总图像数: {total}\n")

print("分辨率统计（按数量排序）：")
for (w, h), cnt in res_counter.most_common():
    print(f"{w}x{h}: {cnt}")