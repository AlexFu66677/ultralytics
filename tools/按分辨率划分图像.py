import os
import shutil
import cv2
from tqdm import tqdm

# ========= 配置 =========
INPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\val\labelme"   # 图像 + txt 同目录
OUTPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\val"

MOVE_FILE = True   # True=移动，False=复制

os.makedirs(OUTPUT_DIR, exist_ok=True)


def move_or_copy(src, dst):
    if MOVE_FILE:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


# ========= 主流程 =========
files = os.listdir(INPUT_DIR)

for file in tqdm(files):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, file)

    # 读取图像尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 读取失败: {file}")
        continue

    h, w = img.shape[:2]

    # 分辨率文件夹
    res_folder = f"{w}x{h}"
    out_dir = os.path.join(OUTPUT_DIR, res_folder)
    os.makedirs(out_dir, exist_ok=True)

    # ===== 移动图像 =====
    dst_img = os.path.join(out_dir, file)
    move_or_copy(img_path, dst_img)

    # ===== 同名txt =====
    base = os.path.splitext(file)[0]
    txt_path = os.path.join(INPUT_DIR, base + ".json")

    if os.path.exists(txt_path):
        dst_txt = os.path.join(out_dir, base + ".json")
        move_or_copy(txt_path, dst_txt)

print("Done!")