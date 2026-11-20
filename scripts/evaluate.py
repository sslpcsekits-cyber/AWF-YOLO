"""Evaluate trained models on the TEST split with identical settings and write the paper's tables.

  python evaluate.py --data /content/data/gc10/gc10.yaml --out results/gc10 \
      --weights runs/gc10_yolo11n/weights/best.pt runs/gc10_awf-yolo/weights/best.pt ...

Writes: summary.csv (mAP@0.5, mAP@0.5:0.95, P, R, F1, params, GFLOPs, checkpoint MB, FP32 parameter MB,
latency and FPS at batch 1), per_class_ap.csv, label_counts.csv (instances per class and split, shares sum
to 100%), and one folder per model with Ultralytics plots (PR curves, confusion matrix)."""
import argparse, copy, csv, glob, os
from collections import Counter
import _path  # noqa: F401
import awf  # noqa: F401
import torch
from ultralytics import RTDETR, YOLO


def gflops(model, imgsz=640):
    try:
        import thop
        m = copy.deepcopy(model).eval().float().cpu()
        macs, _ = thop.profile(m, inputs=(torch.zeros(1, 3, imgsz, imgsz),), verbose=False)
        return 2 * macs / 1e9
    except Exception:  # noqa: BLE001
        return float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--weights", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# label statistics (instance counts in paper Tables 4 and 12)
import yaml
cfg = yaml.safe_load(open(a.data)); root = cfg.get("path", os.path.dirname(a.data))
names = cfg["names"] if isinstance(cfg["names"], dict) else dict(enumerate(cfg["names"]))
with open(os.path.join(a.out, "label_counts.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["split", "class", "instances", "share_%", "images"])
    for sp in ("train", "val", "test"):
        cnt, imgs = Counter(), Counter()
        for lf in glob.glob(os.path.join(root, "labels", sp, "*.txt")):
            c = [int(l.split()[0]) for l in open(lf) if l.strip()]
            cnt.update(c); imgs.update(set(c))
        tot = sum(cnt.values()) or 1
        for k in sorted(names):
            w.writerow([sp, names[k], cnt[k], round(100 * cnt[k] / tot, 2), imgs[k]])
        w.writerow([sp, "TOTAL", tot, 100.0, ""])

rows, per_class = [], []
for wpath in a.weights:
    tag = os.path.basename(os.path.dirname(os.path.dirname(wpath)))
    model = RTDETR(wpath) if "rtdetr" in tag.lower() else YOLO(wpath)
    net = model.model
    n = sum(p.numel() for p in net.parameters())          # count before val() fuses Conv+BN
    gf = gflops(net, a.imgsz)
    m = model.val(data=a.data, split="test", imgsz=a.imgsz, batch=16, conf=0.001, iou=0.7, plots=True,
                  project=a.out, name=tag, exist_ok=True, verbose=False)
    sp = model.val(data=a.data, split="test", imgsz=a.imgsz, batch=1, plots=False, verbose=False,
                   project=a.out, name=tag + "_speed", exist_ok=True).speed
    p, r = float(m.box.mp), float(m.box.mr)
    ap50 = m.box.ap50
    mean_cls = float(ap50.mean())
    if abs(mean_cls - float(m.box.map50)) > 1e-6:
        print(f"WARNING {tag}: mean per-class AP {mean_cls:.4f} != mAP@0.5 {float(m.box.map50):.4f}")
    rows.append(dict(model=tag, mAP50=round(100 * m.box.map50, 1), mAP50_95=round(100 * m.box.map, 1),
                     precision=round(100 * p, 1), recall=round(100 * r, 1),
                     F1=round(2 * p * r / (p + r + 1e-9), 3), params_M=round(n / 1e6, 3),
                     GFLOPs=round(gf, 2), ckpt_MB=round(os.path.getsize(wpath) / 1e6, 2),
                     fp32_param_MB=round(n * 4 / 1e6, 2), preprocess_ms=round(sp["preprocess"], 2),
                     inference_ms=round(sp["inference"], 2), postprocess_ms=round(sp["postprocess"], 2),
                     FPS_inference=round(1000 / sp["inference"], 1),
                     GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    for i, c in enumerate(m.box.ap_class_index):
        per_class.append(dict(model=tag, cls=names[int(c)], AP50=round(100 * float(ap50[i]), 1),
                              AP50_95=round(100 * float(m.box.ap[i]), 1)))
    print(rows[-1])

for fn, data in (("summary.csv", rows), ("per_class_ap.csv", per_class)):
    with open(os.path.join(a.out, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
print("\nWrote", os.path.join(a.out, "summary.csv"), "and per_class_ap.csv, label_counts.csv")
