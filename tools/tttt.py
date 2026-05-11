import os

# 文件夹路径
folder_path = r"C:\Users\fjl\Desktop\240\bird_kite_yolo\yolo"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    # 只处理文件
    if os.path.isfile(old_path):
        new_name = "coco_bk_" + filename
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")

print("完成")