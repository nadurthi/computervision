# Architectures: every method, layer by layer

This document explains how each method in this repo turns sensor data into box instances, down to tensor
shapes, parameter counts and the exact thresholds in the code. All parameter counts, shapes and compute
figures below were measured on the models used in the experiments (`torch` + `thop`), not copied from papers.

Contents

1. [The four families at a glance](#1-the-four-families-at-a-glance)
2. [Shared geometry layer](#2-shared-geometry-layer) (used by every depth-aware method and by the evaluator)
3. [Classical camera methods](#3-classical-camera-methods): RGB lines, depth edges, RGB-D seam valleys
4. [Lidar methods](#4-lidar-methods): range-image cuts, RANSAC + DBSCAN, camera-to-lidar transfer
5. [SeamNet](#5-seamnet-a-small-rgb-d-u-net-that-segments-seams) (trained here)
6. [YOLO11n-seg](#6-yolo11n-seg-one-stage-instance-segmentation) (fine-tuned here)
7. [FastSAM-s](#7-fastsam-s-yolo-style-segment-everything) (zero-shot)
8. [MobileSAM and SAM 2.1-tiny](#8-mobilesam-and-sam-21-tiny-promptable-segmentation) (zero-shot)
9. [Turning "segment everything" into boxes: the depth-prior selector](#9-turning-segment-everything-into-boxes)
10. [From a mask to a grasp: face fitting and pick order](#10-from-a-mask-to-a-grasp)
11. [What each architecture can and cannot see](#11-what-each-architecture-can-and-cannot-see)

---

## 1. The four families at a glance

| Method | Input | What is learned | Params | Compute | Output |
|---|---|---|---:|---:|---|
| C1 RGB lines | RGB | nothing | 0 | – | label map |
| C2 Depth edges | depth | nothing | 0 | – | label map |
| C3 RGB-D seam valleys | RGB + depth | nothing | 0 | – | label map |
| L1 Range-image cuts | lidar scan | nothing | 0 | – | point labels |
| L2 RANSAC + DBSCAN | lidar points | nothing | 0 | – | point labels |
| L3 Camera → lidar | image labels + lidar | (whatever made the image labels) | – | – | point labels |
| SeamNet RGB-D | RGB + inverse depth + normals + validity (8 ch) | everything, on synthetic walls | 1.10 M | 7.2 GMAC @ 320×240 | 3-class map → instances |
| SeamNet RGB | RGB (3 ch) | everything, on synthetic walls | 1.10 M | 7.1 GMAC @ 320×240 | 3-class map → instances |
| YOLO11n-seg | RGB | COCO pre-training, fine-tuned on synthetic walls | 2.88 M | 2.8 GMAC @ 480×480 | boxes + masks |
| FastSAM-s | RGB | SA-1B subset (class-agnostic), not fine-tuned | 11.8 M | 11.3 GMAC @ 480×480 | all masks |
| MobileSAM | RGB + point prompts | SA-1B distillation, not fine-tuned | 10.1 M (6.1 M encoder) | 39 GMAC encoder @ 1024² | 1 mask per prompt |
| SAM 2.1-tiny | RGB + point prompts | SA-1B + SA-V, not fine-tuned | 39.0 M (27.2 M encoder) | 146 GMAC encoder @ 1024² | 1 mask per prompt |

Compute is multiply-accumulates counted by `thop` over conv and linear layers. It does not count the
matrix products inside attention, so the two SAM encoders are lower bounds. Clean CPU runtimes for every method are
in [`results/speed_cpu.json`](../results/speed_cpu.json).

The families differ in where the box prior lives:

- **Classical:** in hand-written rules (a seam is a depth jump, a crease, or a thin dark line; a face is
  planar and rectangular).
- **Trained (SeamNet, YOLO):** in the weights, learned from 1,200 synthetic walls whose labels are exact.
- **Zero-shot (FastSAM, SAM):** in a generic notion of "object", plus a small geometric selector (§9)
  that injects the box prior after the fact.

---

## 2. Shared geometry layer

Code: [`boxwall/geometry.py`](../boxwall/geometry.py)

**Camera model.** Pinhole, 480×360, f<sub>x</sub> = f<sub>y</sub> = 380 px, principal point at the image
centre. A pixel (u, v) with depth z back-projects to

    X = (u − c_x) z / f_x,   Y = (v − c_y) z / f_y,   Z = z

**Hole filling.** The simulated depth sensor drops ~0.4 % of pixels at random and ~50 % of pixels next to depth
discontinuities, which is where real structured-light and ToF sensors lose returns. `fill_depth` fills holes
with a growing max filter (3, 5, 9, 17 px). The max keeps the *far* side of a discontinuity, so a dropout
at a seam is filled with "gap" depth rather than smeared across both boxes.

**Normals.** The point image P(u, v) is box-filtered (5×5) and normals are the cross product of central
differences:

    n = (P(u+1,v) − P(u−1,v)) × (P(u,v+1) − P(u,v−1)),   flipped to face the camera

**Plane fit.** RANSAC with 3-point hypotheses (60 iterations, 8 mm inlier threshold), then an SVD refit on the
inliers. The plane normal is the singular vector with the smallest singular value.

**Face rectangle.** Inlier points are expressed in an in-plane basis (e₁ = camera x projected onto the plane,
e₂ = n × e₁), and `cv2.minAreaRect` gives the face centre, width and height in metres. This one function is
used by the evaluator (pose error), by the pick planner, and by the zero-shot selector.

---

## 3. Classical camera methods

Code: [`boxwall/classical.py`](../boxwall/classical.py)

### C1. RGB straight lines

    RGB → grey → CLAHE (clip 2, 8×8 tiles) → Fast Line Detector (Canny 40/90, length ≥ 25 px)
        → draw lines 2 px thick as "boundary" → distance transform of the non-boundary area
        → seeds = distance > 4 px → watershed on −distance
        → keep regions with area ≥ 500 px, area ≤ 25 % of the image, rectangularity ≥ 0.7

Rectangularity is region area divided by the area of its minimum-area rectangle. The method assumes every
long straight line is a box edge. On a box wall that is false: tape edges, print blocks, label borders and
barcode bars are all long straight lines. That assumption is why this method fails, and the failure is
instructive.

### C2. Depth jumps + normal creases

    depth → fill_depth → normals
    boundary = jump ∪ crease ∪ dropout, dilated 3×3
      jump(u)    : |z(u) − z(u')| > 0.012·z + 4 mm, for u' in 4 neighbour directions
      crease(u)  : angle(n(u), n(u')) > 22°, u' two pixels right / down
      dropout(u) : raw depth == 0
    seeds = connected components of ¬boundary
    labels = watershed(|∇z|, seeds)      grow seeds back across the boundary band
    keep regions with 300 px ≤ area ≤ 25 % image and a plane fit with ≥ 60 % inliers (10 mm)

The jump threshold scales with z because depth noise grows with distance (σ ≈ 1.2 mm · z² in the simulator,
which is typical of active stereo). It was tuned on the validation split (0.008 / **0.012** / 0.02).
Sensor dropouts are treated as boundaries on purpose: they concentrate exactly on discontinuities, so
"no return" is evidence of an edge.

**What it cannot see:** two boxes sitting flush (same depth, same orientation, gap narrower than one depth
pixel, ~5 mm at 2 m) produce no jump and no crease. They come out as one region.

### C3. Depth edges + RGB seam valleys

C2 merges flush boxes, and RGB contains the cue that separates them: the seam between two cartons is a thin
dark line (a shadowed gap, or crushed, darker cardboard at the edges). Tape and print edges are *steps*,
bright on one side and dark on the other, not *valleys*. C3 adds a valley detector:

    grey → Gaussian 3×3 → black-hat (closing − image) with a 9×9 kernel
         → normalise → Fast Line Detector on the black-hat image (length ≥ 30 px)
         → keep lines whose median black-hat response (3×3 max) > 8 grey levels
    for each C2 region: cut along accepted seam lines (extended 6 px at both ends)
         → if ≥ 2 pieces of ≥ 300 px remain, re-grow them to fill the region (watershed)
    re-apply the planarity / size filter

The black-hat responds only to dark structures narrower than the kernel, so it suppresses tape edges (a
step), large dark prints (wider than 9 px) and smooth shading. The threshold was swept on validation
(4–25 grey levels). It is flat between 4 and 8; 8 was kept.

An earlier version scored lines found on the plain grey image with a valley-versus-step profile test. That
barely changed the result: line detectors fire on the *edges* of a dark band rather than its centre, so the
profile was sampled in the wrong place. Detecting lines on the black-hat image fixed this.

---

## 4. Lidar methods

Code: [`boxwall/lidar.py`](../boxwall/lidar.py)

The simulated lidar is a scanning unit mounted 35 cm above the camera: 80 rings over −38°…+30° elevation and
640 columns over ±48° azimuth, with 8 mm range noise and 2 % random dropouts. At 2 m that is about **5 mm
between columns but about 3 cm between rings**. That anisotropy decides what lidar can and cannot resolve.

### L1. Range-image cuts

    scan → range image R (80×640) and point image P (80×640×3)
    fill single dropouts from neighbours (max filter, keeps the far side)
    median 3×3 on R
    jump(u) : |R(u) − R(u')| > 0.015·R + 10 mm, u' = next column, next ring, 2 columns over
    crease  : normals from anisotropic differences (±4 columns, ±1 ring) after a 9×3 blur; angle > 80°
    seeds = components of ¬boundary, opened with a 1×5 element; watershed back over the scan
    drop segments with < 30 points or > 25 % of the scan

With the normal threshold at 25° the segmentation fragmented badly (F1 0.08 on 8 scans), because 8 mm range
noise on 5 mm column spacing makes per-point normals unreliable. Raising it to 80° effectively makes the
method jump-only, which raised F1 to 0.13. The method finds *columns* of boxes and recesses well, and almost
never finds individual boxes inside a flush stack.

### L2. Sequential RANSAC planes + DBSCAN

    repeat up to 80 times on the remaining points:
        plane = Open3D segment_plane(distance 15 mm, 3-point RANSAC, 200 iterations)
        split the plane's inliers with DBSCAN (ε = 35 mm, min 8 points)
        each cluster with 30 … 25 %·N points is one instance; remove the inliers

This is the textbook approach from container-unloading papers. Here it has two failure modes. The whole front
of the wall is one plane to within ±2 cm, so RANSAC takes many boxes at once. Ring spacing (~3 cm) is close
to ε, so DBSCAN bridges any gap narrower than about 3 cm.

### L3. Camera → lidar label transfer

    project every lidar point into the camera:  u = X/Z·f_x + c_x,  v = Y/Z·f_y + c_y
    accept the label at (u, v) only if |z_camera(u, v) − Z| < 4 cm   (the point is the surface the camera saw)

Here lidar is the geometry sensor and the camera supplies instance boundaries. This is the cheapest kind of
fusion and the one used in most deployed unloaders: segment in the image, measure in 3D.

---

## 5. SeamNet: a small RGB-D U-Net that segments seams

Code: [`boxwall/seamnet.py`](../boxwall/seamnet.py), training: [`scripts/train_seamnet.py`](../scripts/train_seamnet.py)

The idea is the one the classical C3 approximates by hand: **predict the seams, not the boxes.** A pixel is
background (trailer or ignored), box interior, or seam. Instances are what is left between seams. This turns
instance segmentation into 3-class semantic segmentation, which a small fully convolutional network does
well and cheaply. The network never has to count boxes or regress masks.

### Input encoding (8 channels at 320×240)

| Channels | Content | Why |
|---|---|---|
| 0–2 | RGB / 255 − 0.5 | appearance: seam darkness, print, tape |
| 3 | 1 / max(z, 0.3 m) − 0.5 | inverse depth: well conditioned near the sensor, bounded far away |
| 4–6 | surface normal (x, y, z) from filled depth | makes creases and yawed faces explicit, so the network doesn't have to differentiate depth itself |
| 7 | fraction of valid raw-depth pixels − 0.5 | dropouts mark discontinuities |

The RGB-only variant uses channels 0–2. Everything else is identical, so the gap between the two is the value
of depth.

### Layers (measured, input 1×8×240×320)

Each block is `[Conv3×3 → BN → ReLU] × 2`. Encoder blocks are joined by 2×2 max-pooling. Decoder blocks take
the bilinearly upsampled lower level concatenated with the encoder skip at the same resolution.

| Block | Input channels | Output shape | Params |
|---|---|---|---:|
| e1 | 8 | 24 × 240 × 320 | 7,008 |
| e2 | 24 | 48 × 120 × 160 | 31,296 |
| e3 | 48 | 96 × 60 × 80 | 124,800 |
| e4 (bottleneck) | 96 | 192 × 30 × 40 | 498,432 |
| d3 | 192 + 96 | 96 × 60 × 80 | 332,160 |
| d2 | 96 + 48 | 48 × 120 × 160 | 83,136 |
| d1 | 48 + 24 | 24 × 240 × 320 | 20,832 |
| head (1×1 conv) | 24 | 3 × 240 × 320 | 75 |
| **total** | | | **1,097,739** (RGB variant 1,096,659) |

Compute is 7.2 GMAC per frame, about 30 % of it in e1 and d1 at full resolution. The bottleneck has stride 8 and
a receptive field of about 70 px at the network input (about 100 px after the decoder), roughly one small box face. That is enough
to see both sides of a seam and decide whether it is a seam or a tape edge. It is not enough to reason about
a whole stack, and it doesn't need to.

**Edge deployment notes:** every operation is Conv/BN/ReLU/max-pool/bilinear-resize/concat, which all map
directly to TensorRT. BN folds into the convolutions, and the weights are 2.2 MB in FP16. Its latency on a
Jetson has not been measured here.

### Targets and loss

    lab  = instance ids resized to 320×240 (nearest)
    seam = pixels within 2 px of a different id (max filter ≠ min filter over 5×5), on boxes only
    target = 0 background/trailer, 1 interior, 2 seam
    loss = cross-entropy with class weights (1, 1, 3)

The seam class is about 18 % of box pixels (measured on the validation set), so it gets 3× weight. A 2 px band at 320×240 is about 3 px at full
resolution, close to the visual width of a real seam.

### Training

AdamW (lr 2·10⁻³, weight decay 10⁻⁴), one-cycle schedule (10 % warm-up), batch 8, 6 epochs over 1,200
synthetic walls. Augmentation: horizontal flip (the normal-x channel is negated with the image), brightness
gain 0.7–1.3, offset ±0.12, per-channel tint ±0.05. About 7 minutes per epoch on 2 CPU cores. The data is cached
once as float16 (1.5 GB).

### Decoding

    interior = softmax[1] > 0.5, opened 3×3; seeds = its connected components (≥ 25 px)
    labels   = watershed(softmax[2] as elevation, seeds, mask = ¬background)
    upsample labels to 480×360 (nearest)

The watershed runs uphill on seam probability, so each seed grows until it meets the crest of a seam. The
mask lets a seed reclaim the seam band on its own side. If a seam is broken for a few pixels, the two
interiors touch and become one seed. That merge is the main failure mode.

---

## 6. YOLO11n-seg: one-stage instance segmentation

Code: Ultralytics 8.4. Fine-tuned from the COCO checkpoint on 700 synthetic walls (one class, "box"), 480×480,
batch 8, 10 epochs with the default Ultralytics augmentation (mosaic, HSV jitter, flips, scale/translate),
about 4.3 minutes per epoch on 2 CPU cores. Validation mask mAP50 was 0.988 (mAP50-95 0.80).

YOLO11n-seg is a detector with a YOLACT-style mask branch: it predicts boxes, and for each box a 32-number
*coefficient vector* that mixes 32 shared *prototype masks*. Shapes below are for a 480×480 input.

### Backbone (stride 2 → 32)

| # | Layer | Output | Params |
|---:|---|---|---:|
| 0 | Conv 3×3 s2 | 16 × 240 × 240 | 464 |
| 1 | Conv 3×3 s2 | 32 × 120 × 120 | 4,672 |
| 2 | C3k2 | 64 × 120 × 120 | 6,640 |
| 3 | Conv 3×3 s2 | 64 × 60 × 60 | 36,992 |
| 4 | C3k2 → **P3** | 128 × 60 × 60 | 26,080 |
| 5 | Conv 3×3 s2 | 128 × 30 × 30 | 147,712 |
| 6 | C3k2 (C3k inner blocks) → **P4** | 128 × 30 × 30 | 87,040 |
| 7 | Conv 3×3 s2 | 256 × 15 × 15 | 295,424 |
| 8 | C3k2 (C3k) | 256 × 15 × 15 | 346,112 |
| 9 | SPPF (3× max-pool 5×5, concat) | 256 × 15 × 15 | 164,608 |
| 10 | C2PSA (position-sensitive attention) → **P5** | 256 × 15 × 15 | 249,728 |

- **C3k2:** a CSP block. Split channels, run a chain of bottlenecks on one half, concatenate, 1×1 fuse. "k2"
  means two inner blocks; in the deeper stages those inner blocks are themselves small CSP blocks (C3k).
- **SPPF:** three chained 5×5 max-pools, so receptive fields of 5, 9 and 13 cells at stride 32. At 480 px input
  that is the whole wall.
- **C2PSA:** a CSP block wrapping multi-head self-attention over the 15×15 = 225 tokens of P5. It is the only
  global-context operator in the network.

### Neck (PAN-FPN)

| # | Layer | Output |
|---:|---|---|
| 11–13 | upsample P5, concat P4, C3k2 | 128 × 30 × 30 |
| 14–16 | upsample, concat P3, C3k2 → **N3** | 64 × 60 × 60 |
| 17–19 | Conv s2, concat (13), C3k2 → **N4** | 128 × 30 × 30 |
| 20–22 | Conv s2, concat (10), C3k2 → **N5** | 256 × 15 × 15 |

Top-down, then bottom-up again, so every output level has both semantics and localisation.

### Segment head (717,680 params with 80 classes; 1 class after fine-tuning)

For each of the 60² + 30² + 15² = **4,725** anchor points it predicts:

- **box:** 4 sides × 16 bins (distribution focal loss, `reg_max = 16`), a discrete distribution over distance
  to each side;
- **class score:** 1 logit after fine-tuning (80 on COCO);
- **mask coefficients:** 32 numbers.

Separately, a **Proto** module on N3 (Conv 3×3 → ConvTranspose 2×2 s2 → Conv 3×3 → Conv 1×1) outputs
**32 prototypes at 120×120** (stride 4). After NMS each surviving box gets its mask:

    mask_k = sigmoid( c_k · Proto ) cropped to box_k,   c_k ∈ ℝ³²,  Proto ∈ ℝ^{32×120×120}

Training assigns anchors to ground truth with the task-aligned assigner (classification score × IoU). Losses
are CIoU + DFL for boxes, BCE for classes, and per-pixel BCE for masks inside each GT box.

**Why it suits boxes, and where it can fail:** a carton face is a near-rectangle, the shape that box regression
and box-cropped masks handle best. Cropping each mask to its own box also stops it from leaking into the
neighbour, which is SeamNet's main failure. In these experiments YOLO was the most accurate method, with the
fewest merges.

The structural risks are still worth knowing for real data:

- masks are assembled at stride 4 from 32 shared prototypes, so a mask edge is only as sharp as those
  prototypes allow;
- NMS can drop one of two adjacent, similar boxes whose predicted boxes overlap heavily, for example strongly
  yawed boxes or a box partly hidden behind another.

It also starts from COCO weights, whereas SeamNet starts from scratch. That is part of why 10 epochs were enough.

---

## 7. FastSAM-s: YOLO-style "segment everything"

FastSAM is a YOLOv8-seg network trained on 2 % of SA-1B as a **single class ("object")**. It has the same
detector-plus-prototypes design as §6, with an older, wider block:

| Part | Details | Params |
|---|---|---:|
| Backbone | Conv s2 ×2, then C2f stages at 64/128/256/512 channels (stride 4…32), SPPF | 5.08 M |
| Neck | PAN-FPN with C2f blocks: 128 × 60², 256 × 30², 512 × 15² at 480 input | 3.94 M |
| Segment head | 3 levels, 1 class, 32 mask coefficients, Proto with 128 hidden channels → 32 × 120 × 120 | 2.77 M |
| **total** | | **11.8 M**, 11.3 GMAC @ 480² |

**C2f** is the YOLOv8 CSP variant: split, run *n* bottlenecks, and concatenate *every* intermediate output
before the 1×1 fuse. It gives richer gradients than C3k2 at the cost of more channels.

"Segment everything" is just running the detector with a low confidence threshold (0.25) and keeping
everything after NMS (IoU 0.7). On a typical wall it returns around 125 masks: whole boxes, box faces, labels, tape strips,
print blocks and groups of boxes, all with the same class. Turning that soup into one mask per carton is the
selector's job (§9).

---

## 8. MobileSAM and SAM 2.1-tiny: promptable segmentation

Both follow the Segment Anything design. A heavy **image encoder** runs once per image and produces a
256 × 64 × 64 embedding. A light **prompt encoder** and **mask decoder** then run once per prompt.

### Image encoders (measured)

**MobileSAM: TinyViT-5M** (6.07 M params, 39 GMAC conv/linear @ 1024², encoder output 256 × 64 × 64)

| Stage | Type | Width | Blocks | Heads | Window |
|---|---|---:|---:|---:|---:|
| patch embed | 2 × Conv s2 | 64 | – | – | – |
| 1 | MBConv (inverted residual) | 64 | 2 | – | – |
| 2 | windowed attention + local conv | 128 | 2 | 4 | 7 |
| 3 | windowed attention + local conv | 160 | 6 | 5 | 14 |
| 4 | windowed attention + local conv | 320 | 2 | 10 | 7 |
| neck | 1×1 conv + LN + 3×3 conv + LN | 256 | – | – | – |

MobileSAM was produced by *decoupled distillation*: TinyViT was trained to reproduce the original SAM ViT-H
image embeddings, and SAM's own prompt encoder and mask decoder were reused unchanged. It inherits SAM's
behaviour with about 1/100 of the encoder parameters (6 M versus 632 M for ViT-H). (The 321 k-parameter classification head in the checkpoint is
unused.)

**SAM 2.1-tiny: Hiera-T + FPN** (27.2 M params, 146 GMAC conv/linear @ 1024²)

| Stage | Blocks | Width | Heads | Attention | Output stride |
|---|---:|---:|---:|---|---:|
| patch embed (7×7 conv s4) | – | 96 | – | – | 4 |
| 1 | 1 | 96 | 1 | windows of 8 | 4 |
| 2 | 2 | 192 | 2 | windows of 4 | 8 |
| 3 | 7 | 384 | 4 | windows of 14; **global** at blocks 5, 7, 9 | 16 |
| 4 | 2 | 768 | 8 | windows of 7 | 32 |
| FPN neck | – | 256 | – | – | returns 256², 128², 64² feature maps |

Hiera is a hierarchical ViT with no convolutions inside the stages. Downsampling is query pooling at the first
block of stages 2–4. Most attention is windowed, and three global-attention blocks in stage 3 give whole-image
context. Unlike the original SAM decoder, SAM 2's decoder also receives the **stride-4 and stride-8 features
from the FPN**, so mask edges are refined from high-resolution features rather than only upsampled from 64×64.
That matters for one-pixel seams.

SAM 2 also has memory attention (5.9 M), a memory encoder (1.4 M) and object-pointer projections for video.
They are not used for single images, so the active image path is about 31.4 M parameters.

### Prompt encoder (6,220 params, shared design)

A point prompt (x, y, label) becomes a 256-d token: a random-Fourier-feature positional encoding of (x, y)
plus a learned embedding for "foreground point". A learned dense positional encoding is added to the image
embedding.

### Mask decoder (MobileSAM 4.06 M, SAM 2.1-t 4.22 M)

    tokens = [IoU token, 4 mask tokens (SAM 2 adds an object-score token)] + prompt tokens
    two-way transformer, 2 layers, 256-d, 8 heads:
        token self-attention → tokens attend to image → MLP → image attends to tokens
    upscale image embedding ×4 with 2 transposed convs  (SAM 2: adds stride-8 / stride-4 FPN features)
    each mask token → 3-layer MLP → dynamic 1×1 filter over the upscaled features → mask logits (256×256)
    IoU token → MLP → predicted IoU for each mask

A single point is ambiguous (a label? the face? the whole box? the stack?), which is why the decoder emits
three masks: roughly part, object and whole. The best-scoring one per prompt is kept.

### How they are run here

There is no detector, so prompts come from a regular **16 × 12 grid of 192 foreground points**. The image is
encoded once (`set_image`), then prompts are decoded in chunks of 16 (decoding all 192 at once at 1024²
needed more than the 8 GB RAM available). This is a reduced version of SAM's "automatic mask generator",
which uses a 32×32 grid, crops and stability filtering.

---

## 9. Turning "segment everything" into boxes

Code: `select()` in [`boxwall/zeroshot.py`](../boxwall/zeroshot.py)

The zero-shot models know what an object is, not what a *carton* is. The selector adds that prior.

    for every candidate mask:
        reject if area < 300 px or > 20 % of the image
        reject if rectangularity < 0.75
        [RGB-D only]
        fit a plane (RANSAC, §2): reject if < 60 % inliers        → one flat face, not a corner or a group
        fit the face rectangle: reject unless 8 cm ≤ width, height ≤ 70 cm   → physically carton-sized
    sort survivors by area, largest first
    paint each into the label map if ≥ 70 % of it is still unclaimed

"Largest first" prefers whole boxes over labels and tape. The plane and size tests reject the multi-box masks
that "largest first" would otherwise also prefer. The size range is the one real prior: it is the range of
cartons the gripper handles, which a deployed system knows. Without depth (the `_rgb` variants), only area and
rectangularity are available, so a stack of two boxes or a large label passes as easily as a single box.

---

## 10. From a mask to a grasp

Code: `pick_plan()` in [`scripts/make_figures.py`](../scripts/make_figures.py)

For every predicted instance:

1. Fit the face plane and rectangle (§2). Keep faces whose normal is within ~30° of the sensor axis
   (n<sub>z</sub> < −0.85); those are suction-pickable from the front.
2. Report the pick as **face centre + normal** (approach direction = −n) and **width × height**, which bounds
   the box and sizes the suction cup array.
3. Order picks **top-down, then nearest-first**. Faces are bucketed by the height of their top edge in 10 cm
   bands, and within a band the smallest z goes first. That never pulls a box from under another.

The evaluation reports the median errors of steps 1–2 against ground truth (normal angle, face-centre
distance, size) for every correctly matched box whose front face is mostly visible.

---

## 11. What each architecture can and cannot see

| Cue that separates two boxes | C1 RGB | C2 depth | C3 RGB-D | L1/L2 lidar | SeamNet RGB-D | YOLO | SAM family |
|---|---|---|---|---|---|---|---|
| depth step (recessed / protruding box) | – | ✓ | ✓ | ✓ | ✓ | from RGB shading only | from RGB shading only |
| orientation change (yaw / tilt crease) | – | ✓ | ✓ | coarse | ✓ | weak | weak |
| thin dark seam between flush boxes | confused with print and tape | – | ✓ (hand-tuned) | – (ring spacing ≫ gap) | ✓ (learned) | ✓ (learned, stride-4 masks) | ✓ if the seam is visible at stride 4–16 |
| repeated identical boxes | – | ✓ | ✓ | ✓ | ✓ | NMS can merge or drop | needs one prompt per box |
| knows tape and print are not edges | – | n/a | partly (black-hat) | n/a | ✓ (learned) | ✓ (learned) | – (often segments labels as objects) |
| metric size prior | – | – | – | – | – | – | only with the §9 selector |

A row with a single ✓ points to a sensor or a prior you cannot drop. On flush walls, the thin-seam row is the
one that matters, and it is a camera cue that lidar at these densities cannot provide.
