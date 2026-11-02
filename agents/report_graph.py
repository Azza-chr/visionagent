import shutil
import os, sys
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


class State(TypedDict, total=False):
    video: str
    stats: dict
    frames: list
    report: str
    decision: str


def analyze(state: State):
    ev = pd.read_csv(f"results/events_{state['video']}.csv")
    counts = ev["class"].value_counts().to_dict()
    helmet, head = counts.get("helmet", 0), counts.get("head", 0)
    compliance = helmet / (helmet + head) if helmet + head else None
    violations = (ev[ev["class"] == "head"]
                  [["track_id", "first_s", "last_s", "duration_s", "mean_conf"]]
                  .round(2).to_dict("records"))
    return {"stats": {"tracks": counts, "compliance": compliance,
                      "violations": violations}}


def search(state: State):
    """Evidence = frame at the middle of each violation track."""
    frames = []
    os.makedirs("results/figures", exist_ok=True)
    for v in state["stats"]["violations"]:
        t = int(round((v["first_s"] + v["last_s"]) / 2))
        src = f"results/frames_all/{state['video']}_{t:03d}s.jpg"
        if os.path.exists(src):
            dst = f"results/figures/{state['video']}_violation_track{int(v['track_id'])}.jpg"
            shutil.copy(src, dst)
            frames.append({"path": dst, "time_s": t})
    return {"frames": frames}


def write_report(state: State):
    s, name = state["stats"], state["video"]
    L = [f"# Safety report: {name}", "", "## Summary"]
    for cls, n in s["tracks"].items():
        L.append(f"- {cls}: {n} stable tracks")
    if s["compliance"] is not None:
        L.append(f"- Helmet compliance (track level): {s['compliance']:.0%}")
    L += ["", "## Violations (worker without helmet)"]
    if not s["violations"]:
        L.append("None detected.")
    for v in s["violations"]:
        L.append(f"- Track {int(v['track_id'])}: {v['first_s']}s to {v['last_s']}s "
                 f"({v['duration_s']}s), mean confidence {v['mean_conf']}")
    if state["frames"]:
        L += ["", "## Key frames (hybrid search)"]
        for f in state["frames"]:
            rel = os.path.relpath(f["path"], "results").replace("\\", "/")
            L.append(f"![t={f['time_s']}s]({rel})")
    L += ["", "## Limitations",
          "- Tracks are not workers: one person can be split into several tracks.",
          "- Vests are not verified (detector misses non-yellow vests).",
          "- Few tracks: percentages are indicative only.",
          "- Violations need human review (context matters)."]
    return {"report": "\n".join(L)}


def human_review(state: State):
    answer = interrupt({"report": state["report"]})      # graph pauses here
    return {"decision": answer}


def finalize(state: State):
    path = f"results/report_{state['video']}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(state["report"])
    print("Saved", path)
    return {}


def route(state: State):
    return "finalize" if state["decision"].strip().lower() == "approve" else END


g = StateGraph(State)
for name, fn in [("analyze", analyze), ("search", search),
                 ("write_report", write_report),
                 ("human_review", human_review), ("finalize", finalize)]:
    g.add_node(name, fn)
g.add_edge(START, "analyze")
g.add_edge("analyze", "search")
g.add_edge("search", "write_report")
g.add_edge("write_report", "human_review")
g.add_conditional_edges("human_review", route, ["finalize", END])
g.add_edge("finalize", END)
app = g.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    video = sys.argv[1]
    cfg = {"configurable": {"thread_id": video}}
    out = app.invoke({"video": video}, cfg)
    if "__interrupt__" in out:
        print(out["__interrupt__"][0].value["report"])
        answer = input("\nApprove this report? (approve / reject): ")
        out = app.invoke(Command(resume=answer), cfg)
    print("Decision:", out.get("decision"))