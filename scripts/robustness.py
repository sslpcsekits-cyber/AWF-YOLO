"""Controlled corruption robustness (paper Table 9): 5 corruption types x 3 severities on the TEST split.
  python robustness.py --data /content/data/gc10/gc10.yaml --work /content/corrupt --out results/gc10 \
      --weights runs/gc10_yolo11n/weights/best.pt runs/gc10_awf-yolo/weights/best.pt"""
import argparse, csv, glob, os, shutil
import _path  # noqa: F401
import awf  # noqa: F401
import cv2, numpy as np, yaml
from ultralytics import RTDETR, YOLO

SEV = {
    "gaussian_noise": [0.05, 0.10, 0.20],      # sigma on [0,1] intensities
    "motion_blur": [3, 7, 11],                 # horizontal kernel length (pixels)
    "brightness": [1.5, 0.7, 0.5],             # multiplicative factor
    "occlusion": [0.10, 0.20, 0.35],           # fraction of image area masked
    "domain_shift": ["histeq", "clahe", "histeq+clahe"],
}


def corrupt(img, kind, lvl, rng):
    if kind == "gaussian_noise":
        return np.clip(img / 255.0 + rng.normal(0, lvl, img.shape), 0, 1) * 255
    if kind == "motion_blur":
        k = np.zeros((lvl, lvl), np.float32); k[lvl // 2, :] = 1 / lvl
        return cv2.filter2D(img, -1, k)
    if kind == "brightness":
        return np.clip(img.astype(np.float32) * lvl, 0, 255)
    if kind == "occlusion":
        h, w = img.shape[:2]; area = lvl * h * w; ar = rng.uniform(0.5, 2.0)
        bh = int(min(h, np.sqrt(area / ar))); bw = int(min(w, area / max(bh, 1)))
        y, x = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
        out = img.copy(); out[y:y + bh, x:x + bw] = 0; return out
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if "histeq" in lvl:
        ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    if "clahe" in lvl:
        ycc[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--weights", nargs="+", required=True)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.data)); root = cfg.get("path", os.path.dirname(a.data))
tests = sorted(glob.glob(os.path.join(root, "images", "test", "*")))
sets = {"clean": a.data}
for kind, levels in SEV.items():
    for i, lvl in enumerate(levels):
        tag = f"{kind}_s{i + 1}"; d = os.path.join(a.work, tag); rng = np.random.default_rng(0)
        if not os.path.exists(os.path.join(d, "done")):
            os.makedirs(f"{d}/images/test", exist_ok=True)
            shutil.copytree(os.path.join(root, "labels", "test"), f"{d}/labels/test", dirs_exist_ok=True)
            for f in tests:
                cv2.imwrite(f"{d}/images/test/{os.path.basename(f)}", corrupt(cv2.imread(f), kind, lvl, rng).astype(np.uint8))
            open(os.path.join(d, "done"), "w").write("ok")
        y = dict(cfg, path=d, train="images/test", val="images/test", test="images/test")
        open(f"{d}/data.yaml", "w").write(yaml.safe_dump(y)); sets[tag] = f"{d}/data.yaml"

rows = []
for w in a.weights:
    name = os.path.basename(os.path.dirname(os.path.dirname(w))); model = RTDETR(w) if "rtdetr" in name.lower() else YOLO(w); clean = None
    for tag, dy in sets.items():
        m = model.val(data=dy, split="test", imgsz=640, batch=16, plots=False, verbose=False)
        v = 100 * float(m.box.map50); clean = v if tag == "clean" else clean
        rows.append(dict(model=name, condition=tag, mAP50=round(v, 1), retention_pct=round(100 * v / clean, 1)))
        print(rows[-1])
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "robustness.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("wrote", os.path.join(a.out, "robustness.csv"))
