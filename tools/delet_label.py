import os
from tqdm import tqdm

# ================= 配置 =================

label_dir = r"C:\Users\fjl\Desktop\240\raw_data\640_640_uav"
img_dir   = r"C:\Users\fjl\Desktop\240\raw_data\640_640_uav"

# 👉 要删除的类别（字符串或数字都行）
DELETE_CLASSES = {"1"}   # 例：{"0"} 或 {"0", "1"}

# 👉 删除策略：
# "any" = 只要包含就删
# "all" = 全部都是这些类别才删
DELETE_MODE = "any"

# 👉 是否删除对应图像
DELETE_IMAGE = True

# 👉 安全模式（True=只打印，不删除）
DRY_RUN = False

# 支持的图像格式
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# =======================================

def should_delete(classes):
    if len(classes) == 0:
        return False

    if DELETE_MODE == "any":
        return any(c in DELETE_CLASSES for c in classes)

    elif DELETE_MODE == "all":
        return all(c in DELETE_CLASSES for c in classes)

    else:
        raise ValueError("DELETE_MODE 必须是 'any' 或 'all'")


deleted = 0
total = 0

label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

for txt_name in tqdm(label_files):
    txt_path = os.path.join(label_dir, txt_name)

    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()
    except:
        print(f"读取失败: {txt_path}")
        continue

    # 提取类别
    classes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 0:
            classes.append(parts[0])

    total += 1

    if should_delete(classes):
        base = os.path.splitext(txt_name)[0]

        # ===== 删除 txt =====
        if DRY_RUN:
            print(f"[TXT] 将删除: {txt_path}")
        else:
            os.remove(txt_path)

        # ===== 删除图像 =====
        if DELETE_IMAGE:
            found_img = False
            for ext in IMG_EXTS:
                img_path = os.path.join(img_dir, base + ext)
                if os.path.exists(img_path):
                    found_img = True
                    if DRY_RUN:
                        print(f"[IMG] 将删除: {img_path}")
                    else:
                        os.remove(img_path)
                    break

            if not found_img:
                print(f"⚠️ 未找到图像: {base}")

        deleted += 1

print("\n===== 统计 =====")
print(f"总标注数: {total}")
print(f"删除数量: {deleted}")
print(f"删除比例: {deleted/total:.2%}" if total > 0 else "0%")