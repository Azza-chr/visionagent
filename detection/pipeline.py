import sys
import cv2
import pandas as pd
from ultralytics import YOLO

name = sys.argv[1]                     # e.g. test2
video = f"data/{name}.mp4"
model = YOLO("models/best.pt")
fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)

rows = []
stream = model.track(video, stream=True, persist=True,
                     tracker="bytetrack.yaml", conf=0.4, verbose=False)
for i, r in enumerate(stream):
    if r.boxes.id is None:
        continue
    for tid, cls, cf in zip(r.boxes.id.int().tolist(),
                            r.boxes.cls.int().tolist(),
                            r.boxes.conf.tolist()):
        rows.append((i, i / fps, tid, model.names[cls], cf))

df = pd.DataFrame(rows, columns=["frame", "time_s", "track_id", "class", "conf"])
events = (df.groupby(["track_id", "class"])
            .agg(first_s=("time_s", "min"), last_s=("time_s", "max"),
                 n_frames=("frame", "count"), mean_conf=("conf", "mean"))
            .reset_index())
events["duration_s"] = (events["last_s"] - events["first_s"]).round(2)
stable = events[events["duration_s"] >= 1.0].copy()
stable.insert(0, "video", name)
stable.to_csv(f"results/events_{name}.csv", index=False)

counts = stable["class"].value_counts()
print(f"\nStable tracks for {name}:\n{counts}")
n_ok, n_bad = counts.get("helmet", 0), counts.get("head", 0)
if n_ok + n_bad > 0:
    print(f"Helmet compliance: {n_ok / (n_ok + n_bad):.0%} ({n_bad} without helmet)")