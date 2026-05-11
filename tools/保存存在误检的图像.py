import os
import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
import shutil
from tqdm import tqdm
# ===================== 配置 =====================
# model_path = r"C:\Users\fjl\Desktop/best.pt"
model_path = "F:\model_zoo\G240\G950_vis_s_260301\weights/best.pt"
img_dir = r"C:\Users\fjl\Desktop\240\train"
label_dir = r"C:\Users\fjl\Desktop\240\train"
save_dir = r"C:\Users\fjl\Desktop\240\out"

names = ["uav", "bird"]

conf_thres = 0.25
iou_thres = 0.3

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/all", exist_ok=True)

# ===== badcase目录 =====
vis_dir = os.path.join(save_dir, "badcase_vis_val")
raw_dir = os.path.join(save_dir, "badcase_raw_val")

os.makedirs(vis_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)

model = YOLO(model_path)

# ===================== 面积分桶 =====================
area_bins = [(0,100),(100,200),(200,300),(300,400),(400,500),
             (500,600),(600,700),(700,800),(800,900),(900,1e10)]

bin_labels = ["<100","100-200","200-300","300-400","400-500",
              "500-600","600-700","700-800","800-900",">900"]

def get_area(box):
    x1,y1,x2,y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

def get_bin_id(area):
    for i,(l,r) in enumerate(area_bins):
        if l <= area < r:
            return i
    return -1

# ===================== GT读取 =====================
KEEP_CLASSES = {
    0: 0,  # uav -> uav
    2: 1   # bird -> bird
}


def load_gt(label_path, img_shape):
    gts = []

    if not os.path.exists(label_path):
        return gts

    h, w = img_shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            cls, x, y, bw, bh = map(float, line.split())
            cls = int(cls)

            if cls not in KEEP_CLASSES:
                continue

            cls = KEEP_CLASSES[cls]

            # === YOLO格式 → xyxy ===
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            gts.append([x1, y1, x2, y2, cls])

    return gts
# ===================== 匹配 =====================
def match_with_yolo(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf):

    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return [], [], [], []

    if len(gt_boxes) == 0:
        fp = [(pred_boxes[i], pred_cls[i], pred_conf[i]) for i in range(len(pred_boxes))]
        return [], fp, [], []

    if len(pred_boxes) == 0:
        fn = [(gt_boxes[i], gt_cls[i]) for i in range(len(gt_boxes))]
        return [], [], fn, []

    iou = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes)).numpy()
    x = np.where(iou > iou_thres)

    if len(x[0]):
        matches = np.concatenate([np.stack(x,1), iou[x[0],x[1]][:,None]], axis=1)

        if len(matches) > 1:
            matches = matches[matches[:,2].argsort()[::-1]]
            matches = matches[np.unique(matches[:,1], return_index=True)[1]]
            matches = matches[matches[:,2].argsort()[::-1]]
            matches = matches[np.unique(matches[:,0], return_index=True)[1]]
    else:
        matches = np.zeros((0,3))

    tp, fp, fn, cls_err = [], [], [], []
    matched_gt, matched_pred = set(), set()

    for gt_i, pred_i, _ in matches:
        gt_i, pred_i = int(gt_i), int(pred_i)
        matched_gt.add(gt_i)
        matched_pred.add(pred_i)

        if gt_cls[gt_i] == pred_cls[pred_i]:
            tp.append((pred_boxes[pred_i], gt_boxes[gt_i],
                       pred_cls[pred_i], gt_cls[gt_i], pred_conf[pred_i]))
        else:
            cls_err.append((pred_boxes[pred_i], gt_boxes[gt_i],
                            pred_cls[pred_i], gt_cls[gt_i], pred_conf[pred_i]))

    for i in range(len(gt_boxes)):
        if i not in matched_gt:
            fn.append((gt_boxes[i], gt_cls[i]))

    for i in range(len(pred_boxes)):
        if i not in matched_pred:
            fp.append((pred_boxes[i], pred_cls[i], pred_conf[i]))

    return tp, fp, fn, cls_err

# ===================== 画框 =====================
def draw_results(img, tp, fp, fn, cls_err, names):
    vis = img.copy()

    # TP 绿
    for p, g, pc, gc, conf in tp:
        x1,y1,x2,y2 = map(int, p)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)
        cv2.putText(vis, f"TP:{names[pc]} {conf:.2f}",
                    (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # FP 红
    for p, pc, conf in fp:
        x1,y1,x2,y2 = map(int, p)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 1)
        cv2.putText(vis, f"FP:{names[pc]} {conf:.2f}",
                    (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # FN 蓝（GT）
    for g, gc in fn:
        x1,y1,x2,y2 = map(int, g)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 1)
        cv2.putText(vis, f"FN:{names[gc]}",
                    (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # 分类错误 黄
    for p, g, pc, gc, conf in cls_err:
        x1,y1,x2,y2 = map(int, p)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,255), 1)
        cv2.putText(vis, f"ERR:{names[pc]}->{names[gc]}",
                    (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    return vis

# ===================== 主流程 =====================
img_paths = list(Path(img_dir).glob("*.jpg"))
badcase_count = 0
pbar = tqdm(img_paths, desc="Processing", ncols=120)

for img_path in pbar:

    img = cv2.imread(str(img_path))

    r = model.predict(img, imgsz=1920, conf=conf_thres, verbose=False)[0]

    if r.boxes is not None and len(r.boxes):
        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_cls = r.boxes.cls.cpu().numpy().astype(int)
        pred_conf = r.boxes.conf.cpu().numpy()
    else:
        pred_boxes, pred_cls, pred_conf = [], [], []

    gts = load_gt(str(Path(label_dir)/(img_path.stem+".txt")), img.shape)
    gt_boxes = [g[:4] for g in gts]
    gt_cls = [g[4] for g in gts]

    tp, fp, fn, cls_err = match_with_yolo(
        gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf
    )

    has_error = (len(fp) > 0) or (len(fn) > 0) or (len(cls_err) > 0)

    if has_error:

        badcase_count += 1

        vis_img = draw_results(img, tp, fp, fn, cls_err, names)

        save_name = img_path.stem

        cv2.imwrite(os.path.join(vis_dir, save_name + ".jpg"), vis_img)
        cv2.imwrite(os.path.join(raw_dir, save_name + ".jpg"), img)

        label_path = Path(label_dir)/(img_path.stem+".txt")
        if label_path.exists():
            shutil.copy(label_path, os.path.join(raw_dir, save_name + ".txt"))

    # ✅ 实时显示信息
    pbar.set_postfix({
        "badcase": badcase_count,
        "fp": len(fp),
        "fn": len(fn),
        "err": len(cls_err)
    })

print("✅ 完成：badcase已保存 + 可视化完成")