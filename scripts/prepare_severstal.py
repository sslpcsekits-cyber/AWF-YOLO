"""Severstal steel defect dataset (Kaggle competition 'severstal-steel-defect-detection') -> YOLO boxes
(one box per connected component of each RLE mask), seed-42 70/15/15 split of defective images.
Optional --max-images to subsample for limited GPU time (state this in the paper).
    python prepare_severstal.py --root /content/severstal --out /content/data/severstal --max-images 4000"""
import argparse, os, random, shutil
import cv2, numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--min-area", type=int, default=30); ap.add_argument("--max-images", type=int, default=0)
ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
H, W = 256, 1600


def rle_to_mask(rle):
    s = list(map(int, rle.split())); m = np.zeros(H * W, np.uint8)
    for st, ln in zip(s[0::2], s[1::2]):
        m[st - 1:st - 1 + ln] = 1
    return m.reshape(W, H).T


df = pd.read_csv(os.path.join(a.root, "train.csv"))
boxes = {}
for img, cls, rle in df[["ImageId", "ClassId", "EncodedPixels"]].itertuples(index=False):
    _, _, stats, _ = cv2.connectedComponentsWithStats(rle_to_mask(rle), connectivity=8)
    for x, y, w, h, area in stats[1:]:
        if area >= a.min_area:
            boxes.setdefault(img, []).append(f"{int(cls)-1} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
imgs = sorted(boxes); random.seed(a.seed); random.shuffle(imgs)
if a.max_images:
    imgs = imgs[:a.max_images]
n = len(imgs); tr, va = int(.7 * n), int(.85 * n)
for sp, lst in {"train": imgs[:tr], "val": imgs[tr:va], "test": imgs[va:]}.items():
    os.makedirs(f"{a.out}/images/{sp}", exist_ok=True); os.makedirs(f"{a.out}/labels/{sp}", exist_ok=True)
    for im in lst:
        shutil.copy(os.path.join(a.root, "train_images", im), f"{a.out}/images/{sp}/{im}")
        open(f"{a.out}/labels/{sp}/{os.path.splitext(im)[0]}.txt", "w").write("\n".join(boxes[im]) + "\n")
    print(sp, len(lst), "images")
open(f"{a.out}/severstal.yaml", "w").write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                           "names:\n  0: defect_1\n  1: defect_2\n  2: defect_3\n  3: defect_4\n")
print("wrote", f"{a.out}/severstal.yaml")
