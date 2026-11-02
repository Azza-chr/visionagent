import csv
import cv2
from ultralytics import YOLO

VIDEO = "data/test.mp4"
model = YOLO("models/best.pt")
fps = cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FPS)

with open("results/detections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time_s", "track_id", "class", "conf", "x1", "y1", "x2", "y2"])
    results = model.track(VIDEO, stream=True, persist=True,
                          tracker="bytetrack.yaml", conf=0.4, verbose=False)
    for i, r in enumerate(results):
        if r.boxes.id is None:
            continue
        for box, tid, cls, cf in zip(r.boxes.xyxy.tolist(),
                                     r.boxes.id.int().tolist(),
                                     r.boxes.cls.int().tolist(),
                                     r.boxes.conf.tolist()):
            writer.writerow([i, round(i / fps, 2), tid, model.names[cls],
                             round(cf, 2), *[round(v) for v in box]])
print("Saved results/detections.csv")