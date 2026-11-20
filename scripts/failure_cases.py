"""Annotated failure cases (paper Fig. 9): false negatives, false positives, wrong class, merged boxes.
  python failure_cases.py --weights runs/gc10_awf-awb/weights/best.pt --data /content/data/gc10/gc10.yaml --out results/paper/failures
Green = ground truth (class), red = prediction (class + confidence). Every failure is listed in failures.csv
(image, type, explanation, focus box) and drawn as a readable crop in <out>/<type>/. figure_failure_cases.jpg is a 2x2
grid with ONE example per type chosen at random with a fixed seed (--seed), not the best or worst case.
Counting rules (conf >= --conf, IoU >= --iou): false negative = ground-truth box matched by no prediction;
false positive = prediction overlapping no ground truth; wrong class = matched box with a different class;
merged = one prediction covering >= 2 ground-truth boxes (>60% of each). Duplicate detections of one object are not counted."""
import argparse, csv, glob, os, random
import cv2, numpy as np, yaml

GREEN, RED = (0, 190, 0), (0, 0, 230)
TW, TH, BAR = 900, 560, 58
TYPES = ["false_negative", "false_positive", "wrong_class", "merged"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def cover(g, p):
    x1, y1, x2, y2 = max(g[0], p[0]), max(g[1], p[1]), min(g[2], p[2]), min(g[3], p[3])
    return max(0, x2 - x1) * max(0, y2 - y1) / ((g[2] - g[0]) * (g[3] - g[1]) + 1e-9)


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def label(img, text, org, color, scale=0.6):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(max(0, min(org[0], img.shape[1] - tw - 4))); y = int(max(th + 4, min(org[1], img.shape[0] - bl - 2)))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + bl), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def tile(img, gts, preds, names, focus, title, why):
    """Crop around `focus`, enlarge, draw boxes with readable labels, add a title bar. Returns (BAR+TH) x TW BGR."""
    h, w = img.shape[:2]
    bw, bh = focus[2] - focus[0], focus[3] - focus[1]
    cw, ch = max(500.0, 2.5 * bw), max(350.0, 2.5 * bh)
    asp = TW / TH
    if cw / ch < asp: cw = ch * asp
    else: ch = cw / asp
    cw, ch = min(cw, w), min(ch, h)
    cx, cy = (focus[0] + focus[2]) / 2, (focus[1] + focus[3]) / 2
    x0 = int(min(max(cx - cw / 2, 0), w - cw)); y0 = int(min(max(cy - ch / 2, 0), h - ch))
    x1, y1 = int(round(x0 + cw)), int(round(y0 + ch))
    crop = img[y0:y1, x0:x1]
    s = min(TW / crop.shape[1], TH / crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    crop = np.ascontiguousarray(crop)
    ch_, cw_ = crop.shape[:2]

    def tr(b): return [(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s]
    def visible(b): return b[2] > 0 and b[3] > 0 and b[0] < cw_ and b[1] < ch_
    for gc, gb in gts:
        b = tr(gb)
        if visible(b):
            cv2.rectangle(crop, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, 2)
            label(crop, f"GT {names[gc]}", (b[0], b[1] - 2 if b[1] > 24 else b[1] + 22), GREEN)
    for pc, pb, ps in preds:
        b = tr(pb)
        if visible(b):
            cv2.rectangle(crop, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), RED, 2)
            label(crop, f"{names[pc]} {ps:.2f}", (b[0], b[3] + 22 if b[3] < ch_ - 26 else b[3] - 4), RED)
    canvas = np.zeros((BAR + TH, TW, 3), np.uint8)
    oy, ox = BAR + (TH - ch_) // 2, (TW - cw_) // 2
    canvas[oy:oy + ch_, ox:ox + cw_] = crop
    cv2.putText(canvas, f"{title}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, why[:95], (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import _path  # noqa: F401
    import awf  # noqa: F401
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(a.data)); root = cfg.get("path", os.path.dirname(a.data))
    model = YOLO(a.weights); names = model.names
    for t in TYPES: os.makedirs(os.path.join(a.out, t), exist_ok=True)
    cases = []
    for path in sorted(glob.glob(os.path.join(root, "images", "test", "*"))):
        img = cv2.imread(path); h, w = img.shape[:2]
        lab = os.path.join(root, "labels", "test", os.path.splitext(os.path.basename(path))[0] + ".txt")
        gts = []
        for l in (open(lab) if os.path.exists(lab) else []):
            if l.strip():
                c, xc, yc, bw, bh = map(float, l.split()[:5])
                gts.append((int(c), [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
        r = model.predict(path, conf=a.conf, verbose=False)[0]
        preds = [(int(c), b.tolist(), float(s)) for c, b, s in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf)]
        found, matched = [], set()
        for pc, pb, ps in preds:
            ious = [iou(pb, gb) for _, gb in gts]
            cov = [j for j, (_, gb) in enumerate(gts) if cover(gb, pb) > 0.6]
            if len(cov) >= 2:
                found.append(("merged", f"one {names[pc]} box (conf {ps:.2f}) covers {len(cov)} ground-truth defects", pb)); matched.update(cov); continue
            if not ious or max(ious) < a.iou:
                found.append(("false_positive", f"{names[pc]} predicted (conf {ps:.2f}) where no defect is annotated", pb)); continue
            j = int(np.argmax(ious)); matched.add(j)
            if gts[j][0] != pc:
                found.append(("wrong_class", f"{names[gts[j][0]]} predicted as {names[pc]} (conf {ps:.2f}, IoU {max(ious):.2f})", union(pb, gts[j][1])))
        for j, (gc, gb) in enumerate(gts):
            if j not in matched: found.append(("false_negative", f"{names[gc]} not detected", gb))
        for t, why, focus in found:
            cases.append(dict(path=path, type=t, why=why, focus=focus, gts=gts, preds=preds))
    rng = random.Random(a.seed)
    chosen = {t: rng.choice([c for c in cases if c["type"] == t]) for t in TYPES if any(c["type"] == t for c in cases)}
    grid, rows = {}, []
    for k, c in enumerate(cases):
        img = cv2.imread(c["path"]); name = os.path.basename(c["path"])
        tl = tile(img, c["gts"], c["preds"], names, c["focus"], f'{c["type"].replace("_", " ")}  |  {name}', c["why"])
        cv2.imwrite(os.path.join(a.out, c["type"], f"{k:04d}_{name}"), tl, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shown = chosen.get(c["type"]) is c
        if shown: grid[c["type"]] = tl
        rows.append([name, c["type"], c["why"], [round(v) for v in c["focus"]], "yes" if shown else ""])
    with open(os.path.join(a.out, "failures.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["image", "type", "explanation", "focus_box", "in_figure"]] + rows)
    blank = np.zeros((BAR + TH, TW, 3), np.uint8)
    t4 = [grid.get(t, blank) for t in TYPES]
    cv2.imwrite(os.path.join(a.out, "figure_failure_cases.jpg"), np.vstack([np.hstack(t4[:2]), np.hstack(t4[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print({t: sum(c["type"] == t for c in cases) for t in TYPES}, "-> see", a.out)
    print("figure examples:", {t: os.path.basename(c["path"]) for t, c in chosen.items()})


if __name__ == "__main__":
    main()
