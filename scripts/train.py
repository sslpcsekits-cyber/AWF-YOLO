"""Train any model in the study under the fixed protocol (awf/trainer.py: PROTOCOL).

  python train.py --model awf-yolo          --data /content/data/gc10/gc10.yaml --name gc10_awf-yolo
  python train.py --model awf-awb           --data ... --name gc10_awf-awb        (ablation)
  python train.py --model awf-yolo@ratio=0.8 --data ... --name gc10_awf-yolo_r0.8  (alpha sensitivity)
  python train.py --model awf-yolo@cls=focal --data ... --name gc10_awf-yolo_focal (class imbalance)
  python train.py --model yolo11n.yaml      --data ... --name gc10_yolo11n         (baseline, from scratch)
  python train.py --model rtdetr-l.pt --batch 8 --pretrained --data ... --name gc10_rtdetr-l
  python train.py --resume /path/to/weights/last.pt                                (after a Colab disconnect)
Results are written to --project (put it on Google Drive so nothing is lost)."""
import argparse
import _path  # noqa: F401
import awf  # noqa: F401  (makes AWF checkpoints loadable)
from awf.trainer import train, AWFTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--data"); ap.add_argument("--name")
ap.add_argument("--project", default="/content/drive/MyDrive/AWF-YOLO-v2/runs")
ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int); ap.add_argument("--seed", type=int)
ap.add_argument("--imgsz", type=int); ap.add_argument("--fraction", type=float)
ap.add_argument("--pretrained", action="store_true", help="use pretrained weights (only for RT-DETR)")
ap.add_argument("--resume")
a = ap.parse_args()

if a.resume:
    import torch
    ck = torch.load(a.resume, map_location="cpu", weights_only=False)
    saved = ck.get("ema") or ck.get("model")
    if isinstance(getattr(saved, "yaml", None), dict) and "awf" in saved.yaml:
        AWFTrainer(overrides=dict(resume=a.resume, model=a.resume)).train()
    else:
        from ultralytics import YOLO
        YOLO(a.resume).train(resume=True)
else:
    over = {k: v for k, v in dict(epochs=a.epochs, batch=a.batch, seed=a.seed, imgsz=a.imgsz,
                                  fraction=a.fraction).items() if v is not None}
    if a.pretrained:
        over["pretrained"] = True
    best = train(a.model, a.data, a.project, a.name, **over)
    print("BEST WEIGHTS:", best)
