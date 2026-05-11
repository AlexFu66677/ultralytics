import cv2
import shutil
from pathlib import Path
import mediapipe as mp

# ===================== 配置 =====================
img_dir = Path(r"C:\Users\fjl\Desktop\240\240\no_label\big_uav\onelabel")
dst_dir = Path(r"C:\Users\fjl\Desktop\240\240\no_label\hands")

# MediaPipe参数
max_num_hands = 2
min_detection_confidence = 0.5

# ==============================================

# 创建输出目录
(dst_dir / "images").mkdir(parents=True, exist_ok=True)
(dst_dir / "labels").mkdir(parents=True, exist_ok=True)

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   # 图片模式
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence
)

# 支持格式
img_exts = [".jpg", ".png", ".jpeg", ".bmp"]

# ===================== 主流程 =====================
for img_path in img_dir.iterdir():
    if img_path.suffix.lower() not in img_exts:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠ 读取失败: {img_path.name}")
        continue

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 推理
    results = hands.process(img_rgb)

    has_hand = results.multi_hand_landmarks is not None

    if has_hand:
        print(f"✔ 检测到手: {img_path.name}")

        # ===== 移动图片 =====
        shutil.move(
            str(img_path),
            str(dst_dir / "images" / img_path.name)
        )

        # ===== 移动 JSON =====
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            shutil.move(
                str(json_path),
                str(dst_dir / "labels" / json_path.name)
            )
        else:
            print(f"⚠ 未找到 JSON: {json_path.name}")

    else:
        print(f"✘ 无手: {img_path.name}")

# 释放资源
hands.close()