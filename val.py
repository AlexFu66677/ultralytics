from binary_val import validate_mapped_detection

results = validate_mapped_detection(
    model_path=r"H:\G240_vis_260208\weights/best.pt",
    data_dir=r"C:\Users\fjl\Desktop\240\240\val",
    imgsz=960,
    batch=16,
    conf=0.3,
    iou=0.7,
    device="0",   # 或 "cpu"
    max_det=300,
    plots=False,
)

print(results)
print("P:", results["precision"])
print("R:", results["recall"])
print("mAP50:", results["map50"])
print("mAP50-95:", results["map50_95"])
