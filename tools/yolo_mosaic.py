import os
import cv2
import random
import numpy as np

# ================== 配置区 ==================
IMG_DIR = r"C:\Users\fjl\Desktop\240\240\no_label\balloon_one\train"
LBL_DIR = r"C:\Users\fjl\Desktop\240\240\no_label\balloon_one\train"
OUT_IMG_DIR = r"C:\Users\fjl\Desktop\240\raw_data\balloon\yolo"
OUT_LBL_DIR = r"C:\Users\fjl\Desktop\240\raw_data\balloon\yolo"

resize_w = 960
resize_h = 540

rows = 2
cols = 2

# ===========================================

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)


def load_yolo_label(path):
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, cx, cy, w, h = map(float, line.split())
            labels.append([int(cls), cx, cy, w, h])
    return labels



def save_yolo_label(path, labels):
    with open(path, "w") as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def mosaic_once(img_files, out_idx):
    big_w = resize_w * cols
    big_h = resize_h * rows

    mosaic_img = np.full((big_h, big_w, 3), 114, dtype=np.uint8)
    mosaic_labels = []

    for idx, img_name in enumerate(img_files):
        r = idx // cols
        c = idx % cols

        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (resize_w, resize_h))

        x_offset = c * resize_w
        y_offset = r * resize_h

        mosaic_img[
            y_offset:y_offset + resize_h,
            x_offset:x_offset + resize_w
        ] = img

        labels = load_yolo_label(lbl_path)

        for cls, cx, cy, w, h in labels:
            bx = cx * resize_w
            by = cy * resize_h
            bw = w * resize_w
            bh = h * resize_h

            bx += x_offset
            by += y_offset

            new_cx = bx / big_w
            new_cy = by / big_h
            new_w = bw / big_w
            new_h = bh / big_h

            mosaic_labels.append([cls, new_cx, new_cy, new_w, new_h])

    # ⭐ 自动生成尺寸字符串
    size_tag = f"{big_w}_{big_h}"

    out_img_name = f"balloon_mosaic_1_{size_tag}_{out_idx:05d}.jpg"
    out_lbl_name = f"balloon_mosaic_1_{size_tag}_{out_idx:05d}.txt"

    cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), mosaic_img)
    save_yolo_label(os.path.join(OUT_LBL_DIR, out_lbl_name), mosaic_labels)


all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
num_per_mosaic = rows * cols

out_idx = 0
random.shuffle(all_imgs)

for i in range(0, len(all_imgs), num_per_mosaic):
    batch = all_imgs[i:i + num_per_mosaic]
    if len(batch) < num_per_mosaic:
        batch += random.choices(all_imgs, k=num_per_mosaic - len(batch))
    mosaic_once(batch, out_idx)
    out_idx += 1