import os
import shutil
import cv2
from tqdm import tqdm

# ========= 配置 =========
INPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\111"
OUTPUT_DIR = r"C:\Users\fjl\Desktop\240\small_obj"

AREA_THRESHOLD = 200  # 像素面积阈值

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========= 判断是否包含小目标 =========
def has_small_object(txt_path, img_w, img_h):
    if not os.path.exists(txt_path):
        return False

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            _, cx, cy, w, h = map(float, parts)

            # 转换为像素面积
            bw = w * img_w
            bh = h * img_h
            area = bw * bh

            if area < AREA_THRESHOLD:
                return True

    return False


# ========= 主流程 =========
files = os.listdir(INPUT_DIR)

for file in tqdm(files):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    txt_path = os.path.join(INPUT_DIR, os.path.splitext(file)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 读取失败: {file}")
        continue

    h, w = img.shape[:2]

    # 判断是否包含小目标
    if has_small_object(txt_path, w, h):
        # 复制图像
        shutil.move(img_path, os.path.join(OUTPUT_DIR, file))

        # 复制标注
        if os.path.exists(txt_path):
            shutil.move(
                txt_path,
                os.path.join(OUTPUT_DIR, os.path.basename(txt_path))
            )

print("Done!")