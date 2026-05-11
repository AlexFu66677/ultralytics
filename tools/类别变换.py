import os
import json
import shutil
from tqdm import tqdm

# ========= 配置 =========
INPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\labelme"
OUTPUT_DIR = r"C:\Users\fjl\Desktop\240\raw_data\big_uav\labelme1"

MODE = "json"  # "yolo" / "json" / "auto"

# 👉 类别定义（统一语义）
CLASS_NAMES = ["uav", "drone"]

# 👉 删除 & 替换（全部用 string）
DELETE_CLASSES = set()
CLASS_MAP = {
    "drone": "uav"
}

COPY_IMAGE = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 映射 =========
NAME2ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID2NAME = {i: name for name, i in NAME2ID.items()}


# ========= 自动读取 JSON（多编码兼容） =========
def load_json_auto(path):
    for enc in ["utf-8", "utf-8-sig", "utf-16", "gbk"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except:
            continue

    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8", errors="ignore"))


# ========= JSON =========
def process_json(path_in, path_out):
    try:
        data = load_json_auto(path_in)
    except Exception as e:
        print(f"❌ JSON读取失败: {path_in} | {e}")
        return False

    keep = False

    # ===== labelme =====
    if "shapes" in data:
        new_shapes = []

        for s in data["shapes"]:
            label = str(s.get("label")).strip().lower()

            if label in DELETE_CLASSES:
                continue

            label = CLASS_MAP.get(label, label)

            s["label"] = label
            new_shapes.append(s)

        data["shapes"] = new_shapes
        keep = len(new_shapes) > 0

    # ===== COCO =====
    elif "annotations" in data:
        new_anns = []

        for ann in data["annotations"]:
            cls_id = ann.get("category_id")

            label = ID2NAME.get(cls_id, None)
            if label is None:
                continue

            if label in DELETE_CLASSES:
                continue

            label = CLASS_MAP.get(label, label)

            if label not in NAME2ID:
                continue

            ann["category_id"] = NAME2ID[label]
            new_anns.append(ann)

        data["annotations"] = new_anns
        keep = len(new_anns) > 0

    # 👉 有标注才写
    if keep:
        with open(path_out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return keep


# ========= YOLO =========
def process_yolo(path_in, path_out):
    new_lines = []

    with open(path_in, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(parts[0])
        except:
            continue

        label = ID2NAME.get(cls_id, None)
        if label is None:
            continue

        label = label.strip().lower()

        if label in DELETE_CLASSES:
            continue

        label = CLASS_MAP.get(label, label)

        if label not in NAME2ID:
            continue

        parts[0] = str(NAME2ID[label])
        new_lines.append(parts)

    if len(new_lines) == 0:
        return False

    with open(path_out, "w") as f:
        for p in new_lines:
            f.write(" ".join(p) + "\n")

    return True


# ========= 工具：复制同名图片 =========
def copy_image(base_name):
    for ext in [".jpg", ".png", ".jpeg"]:
        img_path = os.path.join(INPUT_DIR, base_name + ext)
        if os.path.exists(img_path):
            shutil.copy2(
                img_path,
                os.path.join(OUTPUT_DIR, base_name + ext)
            )
            return True
    return False


# ========= 主流程 =========
total = 0
kept = 0
removed = 0

for file in tqdm(os.listdir(INPUT_DIR)):
    in_path = os.path.join(INPUT_DIR, file)
    out_path = os.path.join(OUTPUT_DIR, file)

    if not os.path.isfile(in_path):
        continue

    keep = False

    # ===== JSON =====
    if MODE in ["json", "auto"] and file.endswith(".json"):
        total += 1
        keep = process_json(in_path, out_path)

        if keep:
            kept += 1
            if COPY_IMAGE:
                copy_image(os.path.splitext(file)[0])
        else:
            removed += 1

    # ===== YOLO =====
    elif MODE in ["yolo", "auto"] and file.endswith(".txt"):
        total += 1
        keep = process_yolo(in_path, out_path)

        if keep:
            kept += 1
            if COPY_IMAGE:
                copy_image(os.path.splitext(file)[0])
        else:
            removed += 1

print("Done!")
print(f"总文件: {total}")
print(f"保留: {kept}")
print(f"删除(空标注): {removed}")