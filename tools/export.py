from ultralytics import YOLO
model = YOLO(r"C:\Users\fjl\Desktop/best.pt")
model.export(format="onnx", imgsz=(720, 1280),opset=12)