from ultralytics import YOLO
# Load a model

# # model = YOLO("F:\G950_ir_640_4class_250709\weights/best.pt")  # load an official model
# model = YOLO(r"D:\code\ultralytics\runs\detect\train70/weights/best.pt")
# data = r"C:\Users\fjl\Desktop\240\240\1280_com_out"
# model.predict(data, save=True, imgsz=960, conf=0.2,line_width = 1,half = True)
from ultralytics.models.sam import SAM3SemanticPredictor,SAM3Predictor

# Initialize predictor with configuration
overrides = dict(
    imgsz = [512,640],
    conf=0.25,
    task="detect",
    mode="predict",
    model="D:/code/sam3.pt",
    half=True,  # Use FP16 for faster inference
    save=True,
)
predictor = SAM3Predictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image(r"C:\Users\fjl\Desktop\111/DJI_0009_1477.jpg")

# Query with multiple text prompts
results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]])
