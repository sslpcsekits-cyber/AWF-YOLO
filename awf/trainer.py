"""Training entry point with ONE fixed protocol shared by every model in the paper."""
import glob
import os
from collections import Counter

from ultralytics.models.yolo.detect import DetectionTrainer

from .model import AWFDetectionModel, parse_variant

# The single training protocol used for every experiment (state it verbatim in Section 3).
PROTOCOL = dict(
    epochs=200, imgsz=640, batch=16, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
    weight_decay=5e-4, warmup_epochs=3, cos_lr=True, close_mosaic=10, patience=0,
    pretrained=False, seed=0, deterministic=True, amp=True, workers=2, plots=True, exist_ok=True,
)


# RT-DETR is fine-tuned from COCO weights with its standard optimiser (state this exception in the paper).
RTDETR_PROTOCOL = dict(optimizer="AdamW", lr0=1e-4, epochs=100, batch=8, pretrained=True)


def inverse_frequency_weights(train_images_dir, nc, cap=10.0):
    """Inverse-frequency class weights (mean ~1) from the YOLO label files of the training split."""
    p = str(train_images_dir).rstrip("/")
    labels_dir = p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    if labels_dir == p and p.endswith("images"):
        labels_dir = p[: -len("images")] + "labels"
    cnt = Counter()
    for f in glob.glob(os.path.join(labels_dir, "*.txt")):
        cnt.update(int(l.split()[0]) for l in open(f) if l.strip())
    if not cnt:
        raise FileNotFoundError(f"No label files found in {labels_dir}")
    total = sum(cnt.values())
    return [min(cap, total / (nc * max(cnt[c], 1))) for c in range(nc)]


class AWFTrainer(DetectionTrainer):
    """DetectionTrainer that builds AWFDetectionModel from a variant string such as 'awf-yolo'."""

    def get_model(self, cfg=None, weights=None, verbose=True, **kwargs):
        spec = parse_variant(cfg if cfg is not None else self.args.model)
        nc = self.data["nc"]
        cw = None
        if spec.get("cls") == "weighted" and "class_weights" not in spec:
            train_dir = self.data["train"] if isinstance(self.data["train"], str) else self.data["train"][0]
            cw = inverse_frequency_weights(str(train_dir), nc)
            print("Inverse-frequency class weights:", [round(w, 2) for w in cw])
        model = AWFDetectionModel(spec, ch=self.data.get("channels", 3), nc=nc, verbose=verbose, class_weights=cw)
        if weights:
            model.load(weights)
        return model


def train(model, data, project, name, **overrides):
    """Train an AWF variant ('awf-...') or any Ultralytics model ('yolo11n.yaml', 'rtdetr-l.pt')."""
    args = {**PROTOCOL, **overrides}
    if str(model).startswith("awf-"):
        trainer = AWFTrainer(overrides=dict(model=model, data=data, project=project, name=name, **args))
        trainer.train()
        return trainer.best
    if str(model).startswith("rtdetr"):
        from ultralytics import RTDETR
        args = {**PROTOCOL, **RTDETR_PROTOCOL, **overrides}
        m = RTDETR(model)
    else:
        from ultralytics import YOLO
        m = YOLO(model)
    m.train(data=data, project=project, name=name, **args)
    return os.path.join(m.trainer.save_dir, "weights", "best.pt")
