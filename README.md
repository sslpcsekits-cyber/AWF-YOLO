# AWF-YOLO: Adaptive Wavelet Bottleneck downsampling for lightweight steel surface defect detection

Code for the paper *Adaptive Wavelet Bottleneck Downsampling for Lightweight Steel Surface Defect Detection:
A Reproducible Study on GC10-DET, NEU-DET and Severstal*. Every number in the paper is produced by
`notebooks/AWF_YOLO_paper_notebook_v3.ipynb` from the CSV files written by `scripts/evaluate.py`.

## Model
**AWF-YOLO** (AWF: adaptive wavelet-based frequency) **= YOLO11n in which the five stride-2 convolutions of the backbone are replaced by an Adaptive Wavelet
Bottleneck (AW-B).** The two stride-2 convolutions of the neck, the C3k2 blocks, SPPF, C2PSA and the detection head
are unchanged. AW-B: one-level Haar DWT (LL, LH, HL, HH all kept) → residual depthwise 3×3 refinement →
input-adaptive per-sub-band gate → 1×1 projection (`awf/modules.py`).

| Paper name | Variant string in the code |
|---|---|
| YOLO11n (baseline) | `yolo11n.yaml` |
| **AWF-YOLO** (AW-B only, the proposed model) | `awf-awb` |
| Full four-component model (AW-B + MRFF + C2PSA-CAA + Inner-SIoU; not retained) | `awf-yolo` |
| Ablation steps | `awf-awb-mrff`, `awf-awb-mrff-caa` |
| Class-imbalance variants | `awf-awb@cls=focal`, `awf-awb@cls=weighted` |
| AW-B controls (Section 5.12): space-to-depth instead of Haar, no gate, no refinement | `awf-awb@down=spd`, `awf-awb@gate=0`, `awf-awb@refine=0` |

`awf/fast.py` provides a numerically identical, faster AW-B forward pass (grouped-convolution Haar transform)
for latency measurement: `from awf.fast import fast_patch; fast_patch(YOLO('best.pt').model)`.

## Training protocol (all models)
Ultralytics 8.4.178; trained from scratch; 200 epochs; SGD (learning rate 0.01 → 0.0001, cosine schedule,
momentum 0.937, weight decay 5e-4); 3 warm-up epochs; batch 16; 640-px input unless stated; mixed precision;
deterministic mode; default Ultralytics augmentation with mosaic off for the last 10 epochs; no early stopping;
seeds 0–4. The checkpoint with the best validation mAP is evaluated once on the test split
(confidence 0.001, NMS IoU 0.7). Exception: RT-DETR-L is fine-tuned from COCO weights for 100 epochs
(AdamW, learning rate 1e-4, batch 8).

## Data
| Dataset | Split (seed 42) | Script |
|---|---|---|
| GC10-DET | 70/15/15 → 1,610 / 345 / 345 images | `scripts/prepare_gc10.py` |
| NEU-DET | 80/10/10 → 1,440 / 180 / 180 images | `scripts/prepare_neu.py` |
| Severstal (4,000-image subset of defective images) | 70/15/15 → 2,800 / 600 / 600 images | `scripts/prepare_severstal.py` |

Severstal masks are converted to one box per 8-connected component with an area of at least 30 pixels.
The datasets are public and are not redistributed here.

## Repository layout
| Path | Purpose |
|---|---|
| `awf/modules.py` | AWB, EUCB, MRFF, CAA, C2PSA_CAA |
| `awf/model.py` | Variant builder (options after `@`, e.g. `awf-awb@cls=focal`) |
| `awf/losses.py` | Inner-SIoU, focal BCE, class-weighted BCE |
| `awf/trainer.py` | The single training protocol |
| `awf/fast.py` | Faster equivalent AW-B inference path |
| `scripts/00_selftest.py` | Builds every variant and checks the components (no dataset needed) |
| `scripts/prepare_*.py` | Dataset conversion and splits |
| `scripts/train.py` | Training, with resume after an interruption |
| `scripts/evaluate.py` | Test-split evaluation: summary, per-class AP, label counts |
| `scripts/robustness.py` | Synthetic-corruption robustness (Table 9) |
| `scripts/failure_cases.py` | Annotated failure cases (Fig. 9) |
| `scripts/edge_benchmark.py` | Optional ONNX benchmark for edge devices (not used in the paper) |
| `notebooks/AWF_YOLO_paper_notebook_v3.ipynb` | Colab notebook: training queues, evaluation, statistics, all tables and figures |

## How to run

### Option A: command line (any machine with a CUDA GPU)
```bash
git clone https://github.com/sslpcsekits-cyber/AWF-YOLO.git
cd AWF-YOLO
pip install -r requirements.txt
python scripts/00_selftest.py            # about 1-2 min, no data needed; must end with "SELF-TEST PASSED"
```
Prepare a dataset (each script writes a YOLO-format folder and a `.yaml` file):
```bash
python scripts/prepare_gc10.py --download --out data/gc10                          # GC10-DET
python scripts/prepare_neu.py --root /path/to/NEU-DET --out data/neu               # NEU-DET (Pascal VOC xml)
python scripts/prepare_severstal.py --root /path/to/severstal --out data/severstal --max-images 4000
```
Train, then evaluate on the test split:
```bash
python scripts/train.py --model yolo11n.yaml --data data/gc10/gc10.yaml --name gc10_yolo11n_s1 --seed 1 --project runs
python scripts/train.py --model awf-awb      --data data/gc10/gc10.yaml --name gc10_awf-awb_s1  --seed 1 --project runs
python scripts/evaluate.py --data data/gc10/gc10.yaml --out results/gc10 \
    --weights runs/gc10_yolo11n_s1/weights/best.pt runs/gc10_awf-awb_s1/weights/best.pt
```
`evaluate.py` writes `summary.csv` (mAP@0.5, mAP@0.5:0.95, precision, recall, parameters, GFLOPs),
`per_class_ap.csv` and `label_counts.csv`. To resume an interrupted run:
`python scripts/train.py --resume runs/<name>/weights/last.pt`. Other examples are at the top of `scripts/train.py`.

### Option B: Google Colab (reproduces every table and figure of the paper)
1. In Google Drive, create the folder `MyDrive/AWF-YOLO-v2/`.
2. Put the code there as `awf_yolo_v2.zip`: a zip containing one folder `awf_yolo_v2/` with `awf/`, `scripts/`
   and this README inside.
3. Put the prepared datasets there as `data/gc10.tar` and `data/neu.tar` (tar archives of the `gc10/` and `neu/`
   folders made by the prepare scripts). Severstal is downloaded by notebook cell S1 (needs a Kaggle API key).
4. Open `notebooks/AWF_YOLO_paper_notebook_v3.ipynb` in Colab on a GPU runtime and run, in order:
   cells 1–4 (setup), 5 (training queue), 6–18 (evaluation and analyses), N0–N8 (input-size experiment),
   S1–S5 (Severstal) and R1–R6 (additional analyses and AW-B controls). The table at the top of the notebook lists which cell produces which table or figure.
   Training cells skip finished runs and resume interrupted ones, so re-run them after a disconnect.

Training time on an NVIDIA T4: about 2.2 h per GC10-DET run, 0.8 h per NEU-DET run and 3.8 h per Severstal run
at 640 px.

## Environment used for the paper
Python 3.13.16, Ultralytics 8.4.178, PyTorch 2.11.0 or 2.14.1 (Colab default at the time of each run), CUDA 13.0,
NVIDIA T4.

## Citation
If you use this code, please cite the paper (citation details will be added on publication).
