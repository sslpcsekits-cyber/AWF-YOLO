"""Convert NEU-DET (Pascal VOC xml annotations, e.g. the Kaggle 'NEU surface defect database') to YOLO
with a seed-42 80/10/10 split.   python prepare_neu.py --root /content/NEU-DET --out /content/data/neu"""
import argparse, glob, os, random, shutil
import xml.etree.ElementTree as ET

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.root, "**", "*.jpg"), recursive=True)}
xmls = sorted(glob.glob(os.path.join(a.root, "**", "*.xml"), recursive=True))
items = []
for x in xmls:
    r = ET.parse(x).getroot()
    stem = os.path.splitext(r.findtext("filename") or os.path.basename(x))[0]
    if stem not in imgs:
        stem = os.path.splitext(os.path.basename(x))[0]
    if stem not in imgs:
        continue
    W = float(r.find("size/width").text); H = float(r.find("size/height").text)
    lines = []
    for o in r.findall("object"):
        name = o.findtext("name").strip().lower().replace(" ", "_")
        if name not in CLASSES:
            raise SystemExit(f"Unknown class '{name}' in {x}")
        bb = o.find("bndbox"); x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"{CLASSES.index(name)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    items.append((stem, lines))
items.sort(); random.seed(a.seed); random.shuffle(items)
n = len(items); tr, va = int(0.8 * n), int(0.9 * n)
for sp, part in {"train": items[:tr], "val": items[tr:va], "test": items[va:]}.items():
    os.makedirs(f"{a.out}/images/{sp}", exist_ok=True); os.makedirs(f"{a.out}/labels/{sp}", exist_ok=True)
    for stem, lines in part:
        shutil.copy(imgs[stem], f"{a.out}/images/{sp}/{stem}.jpg")
        open(f"{a.out}/labels/{sp}/{stem}.txt", "w").write("\n".join(lines) + "\n")
    print(sp, len(part), "images")
open(f"{a.out}/neu.yaml", "w").write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                     "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
print("wrote", f"{a.out}/neu.yaml")
