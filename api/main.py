import os
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.types import Command
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from agents.report_graph import app as report_graph

ctx = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy objects are loaded once, at startup
    ctx["clip"] = SentenceTransformer("clip-ViT-B-32")
    ctx["qdrant"] = QdrantClient(path="results/qdrant_db")
    yield
    ctx["qdrant"].close()


app = FastAPI(title="VisionAgent API", lifespan=lifespan)
app.mount("/frames", StaticFiles(directory="results/frames_all"), name="frames")


def load_events(video: str) -> pd.DataFrame:
    path = f"results/events_{video}.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No events for video '{video}'")
    return pd.read_csv(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/kpis/{video}")
def kpis(video: str):
    ev = load_events(video)
    counts = {k: int(v) for k, v in ev["class"].value_counts().items()}
    helmet, head = counts.get("helmet", 0), counts.get("head", 0)
    violations = (ev[ev["class"] == "head"]
                  [["track_id", "first_s", "last_s", "duration_s", "mean_conf"]]
                  .round(2).to_dict("records"))
    return {
        "video": video,
        "tracks": counts,
        "compliance": helmet / (helmet + head) if helmet + head else None,
        "violations": violations,
    }


@app.get("/timeseries/{video}")
def timeseries(video: str, window: int = 5):
    ev = load_events(video)
    end, rows, t = float(ev["last_s"].max()), [], 0
    while t <= end:
        sub = ev[(ev["first_s"] < t + window) & (ev["last_s"] >= t)]
        c = sub["class"].value_counts().to_dict()
        rows.append({"window_start_s": t,
                     "helmet": int(c.get("helmet", 0)),
                     "head": int(c.get("head", 0)),
                     "vest": int(c.get("vest", 0))})
        t += window
    return rows


@app.get("/search")
def search(q: str, video: str | None = None, head_only: bool = False, limit: int = 5):
    must = []
    if video:
        must.append(models.FieldCondition(key="video",
                                          match=models.MatchValue(value=video)))
    if head_only:
        must.append(models.FieldCondition(key="head", range=models.Range(gt=0)))
    flt = models.Filter(must=must) if must else None

    vec = ctx["clip"].encode(q, normalize_embeddings=True).tolist()
    res = ctx["qdrant"].query_points("frames_pad", query=vec, limit=limit,
                                     query_filter=flt).points
    return [{"score": round(p.score, 3),
             "video": p.payload["video"],
             "time_s": p.payload["time_s"],
             "helmet": p.payload["helmet"],
             "head": p.payload["head"],
             "vest": p.payload["vest"],
             "image_url": "/frames/" + os.path.basename(p.payload["path"])}
            for p in res]


class Decision(BaseModel):
    thread_id: str
    decision: str          # "approve" or "reject"


@app.post("/reports/{video}/draft")
def draft_report(video: str):
    load_events(video)
    thread_id = str(uuid.uuid4())
    out = report_graph.invoke({"video": video},
                              {"configurable": {"thread_id": thread_id}})
    if "__interrupt__" not in out:
        raise HTTPException(status_code=500, detail="Graph did not pause for review")
    return {"thread_id": thread_id,
            "report": out["__interrupt__"][0].value["report"]}


@app.post("/reports/decision")
def decide(d: Decision):
    out = report_graph.invoke(Command(resume=d.decision),
                              {"configurable": {"thread_id": d.thread_id}})
    return {"decision": out.get("decision")}