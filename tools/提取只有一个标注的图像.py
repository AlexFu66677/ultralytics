import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ===================== 路径配置 =====================

src_dir = Path(r"C:\Users\fjl\Desktop\240\240\no_label\balloon")

dst_dir = Path(r"C:\Users\fjl\Desktop\240\240\no_label\balloon_one")

# 支持图像后缀
img_exts = [".jpg", ".jpeg", ".png", ".bmp"]

# ===================== 创建输出目录 =====================

dst_dir.mkdir(parents=True, exist_ok=True)

# ===================== 获取所有 json =====================

json_files = list(src_dir.glob("*.json"))

print(f"找到 json 文件数量: {len(json_files)}")

saved_count = 0

# ===================== 遍历 =====================

for json_path in tqdm(json_files):

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # LabelMe 目标数量
        num_objects = len(data.get("shapes", []))

        # 只保留一个目标
        if num_objects != 1:
            continue

        # ===================== 复制 json =====================

        shutil.copy2(
            str(json_path),
            str(dst_dir / json_path.name)
        )

        # ===================== 查找对应图像 =====================

        stem = json_path.stem

        for ext in img_exts:

            img_path = src_dir / f"{stem}{ext}"

            if img_path.exists():

                shutil.move(
                    str(img_path),
                    str(dst_dir / img_path.name)
                )

                break

        saved_count += 1

    except Exception as e:
        print(f"处理失败: {json_path.name}")
        print(e)

print(f"\n保存完成，共保存 {saved_count} 个单目标样本")