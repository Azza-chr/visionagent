import cv2, os

os.makedirs("results/frames_all", exist_ok=True)
for name in ["test", "test2"]:
    cap = cv2.VideoCapture(f"data/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t = 0
    while t * fps < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = 960 / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
        cv2.imwrite(f"results/frames_all/{name}_{t:03d}s.jpg", frame)
        t += 1
    print(name, "->", t, "frames")