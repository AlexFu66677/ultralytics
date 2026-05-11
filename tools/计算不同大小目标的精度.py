import os
import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ===================== 配置 =====================
model_path = "F:/model_zoo/G240/240_vis_260207.pt"
img_dir = r"C:\Users\fjl\Desktop\240\240\val"
label_dir = r"C:\Users\fjl\Desktop\240\240\val"
save_dir = r"C:\Users\fjl\Desktop\240\out"

names = ["uav", "bird"]

conf_thres = 0.25
iou_thres = 0.3

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/all", exist_ok=True)

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
def load_gt(label_path, img_shape):
    gts = []
    if not os.path.exists(label_path):
        return gts

    h, w = img_shape[:2]
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            x2 = (x + bw/2) * w
            y2 = (y + bh/2) * h
            gts.append([x1,y1,x2,y2,int(cls)])
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

# ===================== 表格保存 =====================
def save_area_table(tp_r, fn_r, tp_p, fp_p, err_p, names, save_path):
    table_data = []
    row_labels = []

    for c in range(len(names)):

        # ===== Precision =====
        row_p = []
        for b in range(10):
            tpv = int(tp_p[c, b])
            fpv = int(fp_p[c, b])
            errv = int(err_p[c, b])
            denom = tpv + fpv + errv

            if denom > 0:
                val = tpv / denom
                row_p.append(f"{tpv}/{denom}={val:.2f}")
            else:
                row_p.append("0/0=0")

        table_data.append(row_p)
        row_labels.append(f"P_{names[c]}")

        # ===== Recall =====
        row_r = []
        for b in range(10):
            tpv = int(tp_r[c, b])
            fnv = int(fn_r[c, b])
            denom = tpv + fnv

            if denom > 0:
                val = tpv / denom
                row_r.append(f"{tpv}/{denom}={val:.2f}")
            else:
                row_r.append("0/0=0")

        table_data.append(row_r)
        row_labels.append(f"R_{names[c]}")

    # ===== 画表 =====
    fig, ax = plt.subplots(figsize=(16, 0.6*len(table_data)+2))
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=bin_labels,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # ===== 高亮 =====
    for i in range(len(table_data)):
        for j in range(len(bin_labels)):
            txt = table_data[i][j]

            try:
                val = float(txt.split("=")[-1])
            except:
                val = 0

            if val < 0.5:
                table[i+1, j].set_facecolor("#ffcccc")
            elif val > 0.8:
                table[i+1, j].set_facecolor("#ccffcc")

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ===================== 主流程 =====================
num_classes = len(names)

# 统计
tp_r = np.zeros((num_classes, 10))
fn_r = np.zeros((num_classes, 10))
tp_p = np.zeros((num_classes, 10))
fp_p = np.zeros((num_classes, 10))
err_p = np.zeros((num_classes, 10))

img_paths = list(Path(img_dir).glob("*.jpg"))

for img_path in img_paths:
    img = cv2.imread(str(img_path))

    r = model.predict(img, imgsz=960, conf=conf_thres, verbose=False)[0]

    if r.boxes is not None and len(r.boxes):
        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_cls = r.boxes.cls.cpu().numpy().astype(int)
        pred_conf = r.boxes.conf.cpu().numpy()
    else:
        pred_boxes, pred_cls, pred_conf = [], [], []

    gts = load_gt(str(Path(label_dir)/(img_path.stem+".txt")), img.shape)
    gt_boxes = [g[:4] for g in gts]
    gt_cls = [g[4] for g in gts]

    tp, fp, fn, cls_err = match_with_yolo(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)

    # ===== 面积分桶统计 =====
    for p, g, pc, gc, conf in tp:
        b_gt = get_bin_id(get_area(g))
        if b_gt >= 0:
            tp_r[gc, b_gt] += 1

        b_pred = get_bin_id(get_area(p))
        if b_pred >= 0:
            tp_p[pc, b_pred] += 1

    for g, gc in fn:
        b = get_bin_id(get_area(g))
        if b >= 0:
            fn_r[gc, b] += 1

    for p, pc, conf in fp:
        b = get_bin_id(get_area(p))
        if b >= 0:
            fp_p[pc, b] += 1

    for p, g, pc, gc, conf in cls_err:
        b_pred = get_bin_id(get_area(p))
        if b_pred >= 0:
            err_p[pc, b_pred] += 1

        b_gt = get_bin_id(get_area(g))
        if b_gt >= 0:
            fn_r[gc, b_gt] += 1

# ===== 计算指标 =====
precision = np.zeros((num_classes, 10))
recall = np.zeros((num_classes, 10))

for c in range(num_classes):
    for b in range(10):
        if tp_r[c,b] + fn_r[c,b] > 0:
            recall[c,b] = tp_r[c,b] / (tp_r[c,b] + fn_r[c,b])

        denom = tp_p[c,b] + fp_p[c,b] + err_p[c,b]
        if denom > 0:
            precision[c,b] = tp_p[c,b] / denom

# ===== 保存结果 =====
save_area_table(
    tp_r,
    fn_r,
    tp_p,
    fp_p,
    err_p,
    names,
    os.path.join(save_dir, "area_metrics_table.png")
)

print("✅ 完成（含面积分桶统计）")