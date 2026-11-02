import sqlite3
import pandas as pd

events = pd.read_csv("results/events.csv")

con = sqlite3.connect("results/visionagent.db")
events.to_sql("events", con, if_exists="replace", index=False)

print("--- Summary per class ---")
print(pd.read_sql("""
    SELECT class,
           COUNT(*)                   AS n_tracks,
           ROUND(AVG(duration_s), 1)  AS avg_duration_s,
           ROUND(AVG(mean_conf), 2)   AS avg_conf
    FROM events
    GROUP BY class
""", con))

print("\n--- Helmets present per 5-second window ---")
print(pd.read_sql("""
    WITH RECURSIVE w(t) AS (
        SELECT 0 UNION ALL SELECT t + 5 FROM w WHERE t < 20
    )
    SELECT w.t AS window_start_s,
           COUNT(e.track_id) AS helmets_present
    FROM w
    LEFT JOIN events e
           ON e.class = 'helmet'
          AND e.first_s < w.t + 5
          AND e.last_s >= w.t
    GROUP BY w.t
""", con))

con.close()