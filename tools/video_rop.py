import cv2

input_path = r"D:\dataset\380_2/2026-04-29_16.03.55_00.mp4"
output_path = r"D:\dataset\380_2/2026-04-29_16.03.55_00_2.mp4"

cap = cv2.VideoCapture(input_path)

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 裁剪尺寸
x1, y1 = 400, 58
x2, y2 = 1900, 852
w, h = x2 - x1, y2 - y1

out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop = frame[y1:y2, x1:x2]
    out.write(crop)

cap.release()
out.release()