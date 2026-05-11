from ultralytics import YOLO
# Load a model

# model = YOLO("F:\G950_ir_640_4class_250709\weights/best.pt")  # load an official model
model = YOLO(r"yolov8n.yaml")
# model = YOLO(r"E:\model_zoo\G950/G950_m_ir_640_4class_120_250722.pt")
if __name__ == '__main__':
     results = model.train(data="t123.yaml", epochs=20, imgsz=960,batch = 8, amp=False)

