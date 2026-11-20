"""Build the GC10-DET YOLO dataset with the fixed seed-42 70/15/15 split (1,610 / 345 / 345 images).

Option A (recommended, reproduces your existing split exactly) - from the Supervisely export:
    python prepare_gc10.py --download --out /content/data/gc10
    (downloads with dataset-tools into /content/data/gc10_raw, then converts)
Option B - you already have the extracted folder (ds/img, ds/ann, meta.json):
    python prepare_gc10.py --extracted /path/to/extracted --out /content/data/gc10
"""
import argparse, glob, json, os, random, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--extracted"); ap.add_argument("--download", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

if a.download:
    import dataset_tools as dtools
    raw = a.out.rstrip("/") + "_raw"
    os.makedirs(raw, exist_ok=True)
    dtools.download(dataset="GC10-DET", dst_dir=raw)
    metas = glob.glob(os.path.join(raw, "**", "meta.json"), recursive=True)
    if not metas:
        raise SystemExit(f"meta.json not found under {raw}; extract the archive and use --extracted")
    a.extracted = os.path.dirname(metas[0])
ex = a.extracted
meta = json.load(open(os.path.join(ex, "meta.json")))
classes = [c["title"] for c in meta["classes"]]
cid = {c: i for i, c in enumerate(classes)}
img_dirs = glob.glob(os.path.join(ex, "**", "img"), recursive=True)
img_dir = img_dirs[0]; ann_dir = os.path.join(os.path.dirname(img_dir), "ann")
images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png")))
random.seed(a.seed); random.shuffle(images)
n = len(images); tr, va = int(0.70 * n), int(0.85 * n)
splits = {"train": images[:tr], "val": images[tr:va], "test": images[va:]}
stats = {}
for sp, files in splits.items():
    os.makedirs(f"{a.out}/images/{sp}", exist_ok=True); os.makedirs(f"{a.out}/labels/{sp}", exist_ok=True)
    nobj = 0
    for f in files:
        ann = json.load(open(os.path.join(ann_dir, f + ".json")))
        W, H = ann["size"]["width"], ann["size"]["height"]
        lines = []
        for o in ann["objects"]:
            if o.get("geometryType") != "rectangle":
                continue
            (x1, y1), (x2, y2) = o["points"]["exterior"]
            x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
            lines.append(f"{cid[o['classTitle']]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        shutil.copy(os.path.join(img_dir, f), f"{a.out}/images/{sp}/{f}")
        open(f"{a.out}/labels/{sp}/{os.path.splitext(f)[0]}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
        nobj += len(lines)
    stats[sp] = (len(files), nobj)
json.dump({"seed": a.seed, **splits}, open(f"{a.out}/split_seed{a.seed}.json", "w"), indent=1)
yaml = f"{a.out}/gc10.yaml"
open(yaml, "w").write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                      "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
for sp, (ni, no) in stats.items():
    print(f"{sp:5s}: {ni} images, {no} objects")
print("classes:", classes); print("wrote", yaml)
