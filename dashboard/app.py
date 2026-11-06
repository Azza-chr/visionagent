import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
PUBLIC_API = os.environ.get("PUBLIC_API_URL", API)   # address your browser can reach

st.set_page_config(page_title="VisionAgent", layout="wide")
st.title("VisionAgent: construction site safety")


def get(path, **params):
    r = requests.get(API + path, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


video = st.sidebar.selectbox("Video", ["test", "test2"])
window = st.sidebar.slider("Time window (seconds)", 2, 10, 5)

try:
    k = get(f"/kpis/{video}")
    ts = pd.DataFrame(get(f"/timeseries/{video}", window=window))
except requests.exceptions.RequestException as e:
    st.error("API not reachable. Is it running on port 8000?")
    st.caption(str(e))
    st.stop()

# --- KPI cards ---
c1, c2, c3 = st.columns(3)
c1.metric("Helmet tracks", k["tracks"].get("helmet", 0))
c2.metric("Without helmet (tracks)", k["tracks"].get("head", 0))
comp = k["compliance"]
c3.metric("Helmet compliance", f"{comp:.0%}" if comp is not None else "n/a")
st.caption("Tracks are not workers: one person can be split into several tracks, "
           "so these figures are indicative.")

# --- Time series ---
st.subheader("Objects present over time")
st.line_chart(ts.set_index("window_start_s")[["helmet", "head", "vest"]])

# --- Violations ---
st.subheader("Violations")
if not k["violations"]:
    st.success("No worker without helmet detected in this video.")
else:
    st.dataframe(pd.DataFrame(k["violations"]), hide_index=True)
    for v in k["violations"]:
        t = int(round((v["first_s"] + v["last_s"]) / 2))
        st.image(f"{PUBLIC_API}/frames/{video}_{t:03d}s.jpg", width=520,
                 caption=f"Track {int(v['track_id'])}, evidence frame at {t}s")

# --- Natural language search ---
st.subheader("Search in the video archive")
q = st.text_input("Describe a moment", "a yellow metal container")
head_only = st.checkbox("Only frames where a worker has no helmet")
if st.button("Search"):
    res = get("/search", q=q, video=video, head_only=head_only, limit=5)
    if not res:
        st.info("No result.")
    cols = st.columns(max(len(res), 1))
    for col, item in zip(cols, res):
        col.image(PUBLIC_API + item["image_url"],
                  caption=f"{item['time_s']}s | score {item['score']}")

# --- Report with human validation ---
st.subheader("Report")
if st.button("Generate draft report"):
    r = requests.post(f"{API}/reports/{video}/draft", timeout=300)
    r.raise_for_status()
    st.session_state["draft"] = r.json()
    st.session_state.pop("decision", None)

if "draft" in st.session_state:
    draft = st.session_state["draft"]
    text = "\n".join(l for l in draft["report"].splitlines()
                     if not l.startswith("!["))
    st.markdown(text)
    a, b = st.columns(2)
    for label, col in (("approve", a), ("reject", b)):
        if col.button(label.capitalize()):
            r = requests.post(f"{API}/reports/decision", timeout=60,
                              json={"thread_id": draft["thread_id"],
                                    "decision": label})
            st.session_state["decision"] = r.json().get("decision")
            del st.session_state["draft"]
            st.rerun()

if st.session_state.get("decision") == "approve":
    st.success("Report approved and saved in results/.")
elif st.session_state.get("decision") == "reject":
    st.warning("Report rejected, nothing saved.")