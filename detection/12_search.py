import sys, cv2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

query = sys.argv[1]
only_head = "--head" in sys.argv
if "--tiles" in sys.argv:
    MODE = "tiles"
elif "--pad" in sys.argv:
    MODE = "pad"
else:
    MODE = "crop"
COLLECTION = f"frames_{MODE}"

clip = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(path="results/qdrant_db")

flt = None
if only_head:
    flt = models.Filter(must=[models.FieldCondition(
        key="head", range=models.Range(gt=0))])

vec = clip.encode(query, normalize_embeddings=True).tolist()
limit = 40 if MODE == "tiles" else 5
res = client.query_points(COLLECTION, query=vec, limit=limit, query_filter=flt).points

seen, best = set(), []
for p in res:                       # keep the best tile of each frame
    if p.payload["path"] in seen:
        continue
    seen.add(p.payload["path"])
    best.append(p)
    if len(best) == 5:
        break

tiles = []
for p in best:
    pl = p.payload
    print(f"{p.score:.3f} | {pl['video']} @ {pl['time_s']}s | "
          f"helmet={pl['helmet']} head={pl['head']} vest={pl['vest']}")
    img = cv2.imread(pl["path"])
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * 240 / h), 240))
    cv2.putText(img, f"{p.score:.2f}", (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    tiles.append(img)

if tiles:
    out = f"results/search_{MODE}_" + query[:30].replace(" ", "_") + ".jpg"
    cv2.imwrite(out, cv2.hconcat(tiles))
    print("Saved", out)
client.close()