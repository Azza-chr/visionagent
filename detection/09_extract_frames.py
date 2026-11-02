import cv2, os

os.makedirs("results/frames", exist_ok=True)
for name, times in {"test": [3, 8, 14], "test2": [2, 4, 8]}.items():
    cap = cv2.VideoCapture(f"data/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = 1024 / max(h, w)                     # resize: 4K images are too heavy
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imwrite(f"results/frames/{name}_{t}s.jpg", frame)
        print("saved", f"{name}_{t}s.jpg")