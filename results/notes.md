## Day 1: First detection with YOLOv8n

**Setup**
- Model: YOLOv8n (nano, pretrained on COCO, 80 classes), running on CPU
- Video: test.mp4, 667 frames, 384x640 resolution
- Inference speed: ~36 ms per image (about 27 FPS), measured on CPU

**Effect of the confidence threshold (`conf`)**
- conf=0.1: detects the most people, but also adds other objects (other classes and likely false detections). High recall, low precision.
- conf=0.4: more conservative. On the first frame it detects only 4 people, so small or partly hidden people are missed.
- conf=0.8: not tested yet (expected: very few boxes, only the most certain ones, many people missed).

**What I learned**
- `conf` is the confidence threshold: the model only keeps detections whose score is at or above it.
- Lower conf means more detections (higher recall, lower precision). Higher conf means fewer but more reliable detections (higher precision, lower recall).
- `classes=[0]` keeps only the "person" class (class 0 in COCO) and ignores everything else.
- The "file not displayed" message in VS Code was just it trying to open a video as text, not an error.

**Next**
- Test conf=0.8 and add my observation above
- Tracking with ByteTrack (persistent IDs) and line counting


## Day 3: PPE fine-tuning
- Model: YOLOv8s, 50 epochs, imgsz 640, batch 16, Colab T4
- Dataset: hat vest (Mirass Workspace), 2,590 images, split ~2.1k / 259 / 259
- ~40 s per epoch on T4
- Epoch 7 validation: mAP50 0.703, mAP50-95 0.355 (still improving)

## Test set evaluation (YOLOv8s, 259 images, 2,415 instances)
- mAP50 0.870, mAP50-95 0.509, P 0.880, R 0.798
- head 0.950 / helmet 0.905 / vest 0.756 (mAP50)
- ~12 ms/image on T4
- Test score slightly above validation: no sign of overfitting to the val set
- Weak class: vest (R 0.641, only 103 instances)


## Error analysis: small objects
- Close workers: vest/helmet detected at 0.86-0.88 confidence
- Distant workers (<~40 px tall): vests missed at imgsz=640, conf=0.4
- Tested: conf 0.25 / imgsz 1280 / tiled inference -> [your findings]

## Violation check (test2.mp4, 3840x2160)
- Pipeline: 6 helmet tracks, 1 head track, compliance 86% (7 tracks: demo, not a statistic)
- Manual check of the head track (t=3.9 s, 7.8 s long): TRUE POSITIVE, worker wears a cap, no helmet
- Caveat: worker is seated near the work area, context decides if it is a real violation
- Missed: small background workers (4K video resized to 640 px)

## VLM experiments (Qwen2-VL-2B on 5 key frames)
- Whole-scene JSON: miscounts workers (5 vs 7), invents violations (flags white helmet), loops (30+ repeated entries) when asked to list everyone
- YOLO facts injected into the prompt: VLM inherits YOLO's errors (vest=0 -> "No PPE") and contradicts correct facts; invents hazards (electrocution, falls)
- Per-person crops: helmet/cap correct (1/1), but vest: always "yes" with invented colors (brown, yellow) -> yes-bias; cannot verify absence
- Decision: YOLO produces safety events; VLM only describes the scene (labeled AI-generated)
- Prompt lesson: placeholders like "..." get copied into the answer

## CLIP + Qdrant search (31 frames, 1 per second, CLIP ViT-B/32)
- Plain CLIP search (crop or pad): weak. Chair/phone queries return only `test` frames, no `test2` frame, similarity scores almost flat (0.286 to 0.277)
- Padding vs center-crop: no clear gain (phone query: 2 test2 frames appear with pad, scores 0.191-0.194)
- Negation ("worker without a helmet"): CLIP alone returns 2/5 frames with a real head; with YOLO filter head>0 it returns 5/5 (by construction)
- Takeaway: hybrid search (detector facts as filters + embeddings for meaning) beats embeddings alone
- Next: tiles, to see if zooming recovers small objects


## CLIP ViT-B/32 + Qdrant search results (31 frames)
- Scene-level query ("workers near scaffolding"): high scores (0.31-0.32), plausible results
- Detail/action queries ("sitting on a chair", "talking on the phone"): flat low scores (0.22-0.29), results not clearly relevant, unchanged by padding or tiling
- Negation ("without a helmet"): CLIP alone 2/5; with YOLO filter head>0: 5/5
- Conclusion: use CLIP for scene semantics, YOLO facts as filters (hybrid search); stronger model (SigLIP) as future work


## Search diagnostic
- "a yellow metal container" -> 5/5 test2 frames; "stacked grey concrete blocks" -> 5/5 test2
- CLIP ViT-B/32 retrieves scene/object content reliably, but not actions or fine details (sitting, phone)
- Padding and tiling did not fix action queries; hypothesis "small objects" not supported
- Best design: hybrid search (YOLO facts as filters + CLIP for scene semantics)