import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# ========= 配置 =========
INPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\yolo\train"
OUTPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\yolo\train1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PAD_COLOR = (144, 144, 144)


# ========= 工具 =========
def read_yolo(txt_path):
    labels = []
    if not os.path.exists(txt_path):
        return labels

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            labels.append(parts)
    return labels


def save_yolo(txt_path, labels):
    with open(txt_path, "w") as f:
        for l in labels:
            cls = int(l[0])

            cx, cy, bw, bh = l[1:]

            f.write(
                f"{cls} "
                f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )


# ========= 核心处理 =========
def process_image(img, labels):
    h, w = img.shape[:2]

    target_w, target_h = 640, 480
    # ===== 情况1：不变 =====
    if (w == 640 and h == 480) or (w == 640 and h == 480):
        return img, labels

    # ===== 情况2：小图 → padding =====
    if w <= 1280 and h <= 720:
        canvas = np.full((target_h, target_w, 3), PAD_COLOR, dtype=np.uint8)

        x_offset = (target_w - w) // 2
        y_offset = (target_h - h) // 2

        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

        new_labels = []
        for cls, cx, cy, bw, bh in labels:
            # 反归一化
            cx *= w
            cy *= h
            bw *= w
            bh *= h

            # 平移
            cx += x_offset
            cy += y_offset

            # 归一化
            cx /= target_w
            cy /= target_h
            bw /= target_w
            bh /= target_h

            new_labels.append([cls, cx, cy, bw, bh])

        return canvas, new_labels

    # ===== 情况3：其他 → letterbox =====
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full((target_h, target_w, 3), PAD_COLOR, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    new_labels = []
    for cls, cx, cy, bw, bh in labels:
        # 原像素
        cx *= w
        cy *= h
        bw *= w
        bh *= h

        # 缩放
        cx *= scale
        cy *= scale
        bw *= scale
        bh *= scale

        # 平移
        cx += x_offset
        cy += y_offset

        # 归一化
        cx /= target_w
        cy /= target_h
        bw /= target_w
        bh /= target_h

        new_labels.append([cls, cx, cy, bw, bh])

    return canvas, new_labels


# ========= 主流程 =========
for file in tqdm(os.listdir(INPUT_DIR)):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    txt_path = os.path.join(INPUT_DIR, os.path.splitext(file)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 读取失败: {file}")
        continue

    labels = read_yolo(txt_path)

    new_img, new_labels = process_image(img, labels)

    # 保存
    out_img_path = os.path.join(OUTPUT_DIR, file)
    out_txt_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".txt")

    cv2.imwrite(out_img_path, new_img)

    if len(new_labels) > 0:
        save_yolo(out_txt_path, new_labels)
    elif os.path.exists(txt_path):
        open(out_txt_path, "w").close()

print("Done!")