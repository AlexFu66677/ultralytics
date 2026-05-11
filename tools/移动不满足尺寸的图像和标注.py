import os
import cv2
import shutil
from tqdm import tqdm

# ========= 配置 =========
INPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\yolo\train"
OUTPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\yolo\train1"

TARGET_W = 640
TARGET_H = 480

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


# ========= 主流程 =========
files = os.listdir(INPUT_DIR)

for file in tqdm(files):
    if not file.lower().endswith(IMG_SUFFIX):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    txt_path = os.path.join(INPUT_DIR, os.path.splitext(file)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 读取失败: {file}")
        continue

    h, w = img.shape[:2]

    # ✅ 不满足目标尺寸 → 移动
    if not (w == TARGET_W and h == TARGET_H):

        # ===== 移动图像 =====
        dst_img = os.path.join(OUTPUT_DIR, file)
        shutil.move(img_path, dst_img)

        # ===== 移动标注（如果存在）=====
        if os.path.exists(txt_path):
            dst_txt = os.path.join(OUTPUT_DIR, os.path.basename(txt_path))
            shutil.move(txt_path, dst_txt)

print("Done!")