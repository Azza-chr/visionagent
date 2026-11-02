import cv2
import pandas as pd
from ultralytics import YOLO

ev = pd.read_csv("results/events_test2.csv")
heads = ev[ev["class"] == "head"]

model = YOLO("models/best.pt")
cap = cv2.VideoCapture("data/test2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

for _, row in heads.iterrows():
    t = (row.first_s + row.last_s) / 2          # middle of the track
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if ok:
        annotated = model.predict(frame, conf=0.4, verbose=False)[0].plot()
        path = f"results/violation_track{int(row.track_id)}.jpg"
        cv2.imwrite(path, annotated)
        print(f"Saved {path} (t={t:.1f}s, duration={row.duration_s}s)")