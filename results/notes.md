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