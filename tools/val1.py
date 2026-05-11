from ultralytics import YOLO
# Load a model

model = YOLO(r"H:\G560_26s_9classs\weights/best.pt") # load an official model+
# model = YOLO(r"yolov8n.yaml")
# model = YOLO(r"E:\model_zoo\G950/G950_m_ir_640_4class_120_250722.pt")
if __name__ == '__main__':
     metrics = model.val(data='t123.yaml', imgsz=1280, batch=32, conf=0.3,iou=0.3, device="0")

