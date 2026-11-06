import glob, os, re, sys
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from ultralytics import YOLO

MODE = sys.argv[1] if len(sys.argv) > 1 else "crop"     # "crop" or "pad"
COLLECTION = f"frames_{MODE}"

def prep(img):
    if MODE != "pad":
        return img                                       # CLIP will center-crop it
    s = max(img.size)
    canvas = Image.new("RGB", (s, s), (114, 114, 114))   # gray square
    canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return canvas

paths = sorted(glob.glob("results/frames_all/*.jpg"))
clip = SentenceTransformer("clip-ViT-B-32")
yolo = YOLO("models/best.pt")

images = [prep(Image.open(p).convert("RGB")) for p in paths]
vectors = clip.encode(images, batch_size=16, normalize_embeddings=True,
                      show_progress_bar=True)

url = os.environ.get("QDRANT_URL")
client = QdrantClient(url=url) if url else QdrantClient(path="results/qdrant_db")
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    COLLECTION,
    vectors_config=models.VectorParams(size=vectors.shape[1],
                                       distance=models.Distance.COSINE))

points = []
for i, (p, v) in enumerate(zip(paths, vectors)):
    video, t = re.match(r"(.+)_(\d+)s\.jpg", os.path.basename(p)).groups()
    r = yolo.predict(p, conf=0.4, verbose=False)[0]
    names = [yolo.names[int(c)] for c in r.boxes.cls]
    payload = {"video": video, "time_s": int(t), "path": p,
               "helmet": names.count("helmet"),
               "head": names.count("head"),
               "vest": names.count("vest")}
    points.append(models.PointStruct(id=i, vector=v.tolist(), payload=payload))

client.upsert(COLLECTION, points=points)
print(len(points), "frames indexed in", COLLECTION)
client.close()