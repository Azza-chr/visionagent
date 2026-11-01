from ultralytics import YOLO

model = YOLO("models/best.pt")

for conf, size in [(0.25, 640), (0.25, 1280)]:
    model.predict(
        "data/test.mp4",
        save=True,
        conf=conf,
        imgsz=size,
        project="results",
        name=f"ppe_conf{conf}_img{size}",
    )