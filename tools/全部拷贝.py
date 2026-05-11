import os
import shutil
from tqdm import tqdm

SRC_DIR = r"C:\Users\fjl\Desktop\240\train_data"
DST_DIR = r"C:\Users\fjl\Desktop\240\train"

os.makedirs(DST_DIR, exist_ok=True)

def get_all_files(root):
    file_list = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            file_list.append(os.path.join(dirpath, f))
    return file_list


all_files = get_all_files(SRC_DIR)

for src_path in tqdm(all_files):
    filename = os.path.basename(src_path)
    dst_path = os.path.join(DST_DIR, filename)

    # ✅ 防止重名覆盖
    if os.path.exists(dst_path):
        name, ext = os.path.splitext(filename)
        count = 1
        while True:
            new_name = f"{name}_{count}{ext}"
            new_dst = os.path.join(DST_DIR, new_name)
            if not os.path.exists(new_dst):
                dst_path = new_dst
                break
            count += 1

    shutil.copy2(src_path, dst_path)

print("Done!")