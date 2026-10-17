from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # se télécharge automatiquement au premier lancement
results = model("data/test.mp4", save=True, conf=0.1, classes=[0])
print("Done! Check the runs/detect/ folder")
