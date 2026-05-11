from ultralytics import YOLO
model = YOLO("yolov8s.yaml")
# model = YOLO(r"E:\model_zoo\G950/G950_m_ir_640_4class_120_250722.pt")
if __name__ == '__main__':
     results = model.train(data="uav.yaml", epochs=20, imgsz=960,batch = 8, amp=False)

