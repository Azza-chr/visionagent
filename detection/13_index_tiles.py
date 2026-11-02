import glob, os, re
from PIL import Image
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from ultralytics import YOLO

def pad_square(img):
    s = max(img.size)
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return canvas

paths = sorted(glob.glob("results/frames_all/*.jpg"))
clip = SentenceTransformer("clip-ViT-B-32")
yolo = YOLO("models/best.pt")

images, payloads = [], []
for p in paths:
    img = Image.open(p).convert("RGB")
    W, H = img.size
    video, t = re.match(r"(.+)_(\d+)s\.jpg", os.path.basename(p)).groups()
    r = yolo.predict(p, conf=0.4, verbose=False)[0]
    names = [yolo.names[int(c)] for c in r.boxes.cls]
    base = {"video": video, "time_s": int(t), "path": p,
            "helmet": names.count("helmet"), "head": names.count("head"),
            "vest": names.count("vest")}
    tw, th = int(0.6 * W), int(0.6 * H)
    boxes = [(0, 0, W, H)] + [(x, y, x + tw, y + th)
                              for y in (0, H - th) for x in (0, W - tw)]
    for k, b in enumerate(boxes):
        images.append(pad_square(img.crop(b)))
        payloads.append({**base, "tile": k})

vectors = clip.encode(images, batch_size=16, normalize_embeddings=True,
                      show_progress_bar=True)

client = QdrantClient(path="results/qdrant_db")
if client.collection_exists("frames_tiles"):
    client.delete_collection("frames_tiles")
client.create_collection(
    "frames_tiles",
    vectors_config=models.VectorParams(size=vectors.shape[1],
                                       distance=models.Distance.COSINE))
client.upsert("frames_tiles", points=[
    models.PointStruct(id=i, vector=v.tolist(), payload=pl)
    for i, (v, pl) in enumerate(zip(vectors, payloads))])
print(len(payloads), "tiles indexed")
client.close()