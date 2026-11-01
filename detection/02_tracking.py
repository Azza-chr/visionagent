import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Ligne de comptage : ta vidéo fait 640 de large et 384 de haut,
# y=200 correspond à peu près au milieu
line = sv.LineZone(start=sv.Point(0, 200), end=sv.Point(640, 200))

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
line_annotator = sv.LineZoneAnnotator()

def callback(frame, index):
    # .track() = détection + suivi ByteTrack intégré à YOLO
    result = model.track(
        frame,
        persist=True,               # garde la mémoire entre les images
        tracker="bytetrack.yaml",   # algorithme de suivi
        classes=[0],
        conf=0.4,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)

    # S'il n'y a aucune personne sur l'image, il n'y a pas de numéros
    if detections.tracker_id is None:
        return line_annotator.annotate(frame.copy(), line)

    line.trigger(detections)        # compte les passages

    labels = [f"#{tid}" for tid in detections.tracker_id]
    frame = box_annotator.annotate(frame.copy(), detections)
    frame = label_annotator.annotate(frame, detections, labels=labels)
    frame = line_annotator.annotate(frame, line)
    return frame

sv.process_video("data/test.mp4", "results/tracked.mp4", callback)
print(f"Entrées : {line.in_count} | Sorties : {line.out_count}")