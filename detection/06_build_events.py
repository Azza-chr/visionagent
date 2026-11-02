import pandas as pd

df = pd.read_csv("results/detections.csv")

events = (
    df.groupby(["track_id", "class"])
      .agg(first_s=("time_s", "min"),
           last_s=("time_s", "max"),
           n_frames=("frame", "count"),
           mean_conf=("conf", "mean"))
      .reset_index()
)
events["duration_s"] = (events["last_s"] - events["first_s"]).round(2)
events["mean_conf"] = events["mean_conf"].round(2)

print("Raw tracks:", len(events))
stable = events[events["duration_s"] >= 1.0]
print("Tracks lasting >= 1 s:", len(stable))
print(stable.groupby("class")["track_id"].count())
print("\nLongest tracks:\n", stable.sort_values("duration_s", ascending=False).head(5))

stable.to_csv("results/events.csv", index=False)
print("Saved results/events.csv")