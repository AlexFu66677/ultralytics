import os
import json
from PIL import Image


def txt_to_labelme(txt_file, image_folder, output_folder):
    # 从TXT文件中获取图像文件名
    if os.path.exists(os.path.join(image_folder, os.path.splitext(os.path.basename(txt_file))[0] + '.png')):
        image_filename = os.path.splitext(os.path.basename(txt_file))[0] + '.png'
        image_path = os.path.join(image_folder, os.path.splitext(os.path.basename(txt_file))[0] + '.png')
        print(f"Image file '{image_path}' already exists.")
    elif os.path.exists(os.path.join(image_folder, os.path.splitext(os.path.basename(txt_file))[0] + '.jpg')):
        # 如果图片文件不存在，将后缀从'.png'更改为'.jpg'
        image_filename = os.path.splitext(os.path.basename(txt_file))[0] + '.jpg'
        image_path = os.path.join(image_folder, image_filename)
        print(f"Image file '{image_path}' exist, creating it with .jpg extension.")
    elif os.path.exists(os.path.join(image_folder, os.path.splitext(os.path.basename(txt_file))[0] + '.bmp')):
        # 如果图片文件不存在，将后缀从'.png'更改为'.jpg'
        image_filename = os.path.splitext(os.path.basename(txt_file))[0] + '.bmp'
        image_path = os.path.join(image_folder, image_filename)
        print(f"Image file '{image_path}' exist, creating it with .jpeg extension.")

    # 构建图像文件路径
    # image_path = os.path.join(image_folder, image_filename)？

    # 获取图像的长宽
    image = Image.open(image_path)
    image_width, image_height = image.size

    # 构建LabelMe数据结构
    labelme_data = {
        "version": "5.3.0",
        "flags": {},
        "shapes": [],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "imagePass": None
    }

    with open(txt_file, 'r') as f:
        for line in f:
            # x, y, width, height, score, category_id, truncation, occlusion = map(int, line.strip().split(','))
            numbers_str = line.split()
            numbers_float = [float(num_str) for num_str in numbers_str]
            x, y, width, height = numbers_float[-4:]
            xmin = x * image_width - width * image_width / 2
            ymin = y * image_height - height * image_height / 2
            xmax = x * image_width + width * image_width / 2
            ymax = y * image_height + height * image_height / 2
            category_mapping = {
                # 0: "sandbags",
                # 1: "pipeline",
                # 2: "tent"
                0: "uav",
                1: "plane",
                2: "bird",
                3: "balloon",
                4: "kite",
                # 5: "pipeline",
                # 6: "tank",
                # 7: "tree",
                # 8: "building",
                # 9: "build"
            }
            label = category_mapping.get(int(numbers_float[0]))
            # label = category_mapping.get(int(category_id), "unknown")
            if xmin > image_width or xmin < 0 or xmax > image_width or xmax < 0 or ymin > image_height or ymin < 0 or ymax > image_height or ymax < 0:
                continue
            shape = {
                "label": label,
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            labelme_data["shapes"].append(shape)

    # 将LabelMe数据转换为JSON
    json_data = json.dumps(labelme_data, indent=2)

    # 构建输出JSON文件路径
    output_json_filename = os.path.splitext(os.path.basename(txt_file))[0] + '.json'
    output_json_path = os.path.join(output_folder, output_json_filename)

    # 保存JSON数据到文件
    with open(output_json_path, 'w') as json_file:
        json_file.write(json_data)


import os
import shutil

def batch_convert_folder(txt_folder, image_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历TXT文件夹中的所有文件
    for filename in os.listdir(txt_folder):
        if filename.endswith('.txt'):
            txt_file_path = os.path.join(txt_folder, filename)
            txt_to_labelme(txt_file_path, image_folder, output_folder)
            base_name = os.path.splitext(filename)[0]
            found_img = False
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                img_path = os.path.join(image_folder, base_name + ext)
                if os.path.exists(img_path):
                    shutil.copy2(
                        img_path,
                        os.path.join(output_folder, base_name + ext)
                    )
                    found_img = True
                    break

            if not found_img:
                print(f"⚠️ 未找到图片: {base_name}")


# 使用示例
txt_folder_path = r'C:\Users\fjl\Desktop\240\raw_data\balloon\yolo'
image_folder_path = r'C:\Users\fjl\Desktop\240\raw_data\balloon\yolo'
output_folder_path = r'C:\Users\fjl\Desktop\240\raw_data\balloon\labelme'
batch_convert_folder(txt_folder_path, image_folder_path, output_folder_path)
