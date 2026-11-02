import pandas as pd

df = pd.read_csv("results/detections.csv")
print("Rows:", len(df))
print("\nDetections per class:\n", df["class"].value_counts())
print("\nUnique tracked objects per class:\n", df.groupby("class")["track_id"].nunique())
print("\nDetections per second (first 10 s):\n", df[df.time_s < 10].groupby(df.time_s.astype(int)).size())