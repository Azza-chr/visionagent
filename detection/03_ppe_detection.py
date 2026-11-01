from ultralytics import YOLO

model = YOLO("models/best.pt")
model.predict("data/test.mp4", save=True, conf=0.4, project="results", name="ppe_demo")
print("Done: see results/ppe_demo/")