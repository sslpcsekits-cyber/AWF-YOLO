"""AWF-YOLO detection model built on Ultralytics' DetectionModel (n-scale, YOLO11n layout).

Variant strings (used as `model=` for training):
    awf-base           YOLO11n layout built by this file (sanity check; should match YOLO11n closely)
    awf-awb            + AWB downsampling
    awf-mrff           + MRFF neck only
    awf-awb-mrff       + AWB + MRFF
    awf-awb-mrff-caa   + AWB + MRFF + C2PSA-CAA
    awf-yolo           full model: AWB + MRFF + C2PSA-CAA + Inner-SIoU
Options are appended after '@', e.g.  awf-yolo@ratio=0.8   awf-yolo@cls=focal   awf-yolo@k=7
AW-B controls: awf-awb@down=spd   awf-awb@gate=0   awf-awb@refine=0   (see DEFAULTS below)
Naming: the paper's "AWF-YOLO" (AWF = adaptive wavelet-based frequency) is the variant awf-awb; the
variant awf-yolo is the paper's "full four-component model".
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2PSA, C3k2, SPPF, Concat, Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from .losses import FocalBCE, WeightedBCE, enable_inner_siou
from .modules import AWB, MRFF, C2PSA_CAA

VARIANTS = {
    "awf-base":         dict(awb=False, mrff=False, caa=False, inner=False),
    "awf-awb":          dict(awb=True,  mrff=False, caa=False, inner=False),
    "awf-mrff":         dict(awb=False, mrff=True,  caa=False, inner=False),
    "awf-awb-mrff":     dict(awb=True,  mrff=True,  caa=False, inner=False),
    "awf-awb-mrff-caa": dict(awb=True,  mrff=True,  caa=True,  inner=False),
    "awf-yolo":         dict(awb=True,  mrff=True,  caa=True,  inner=True),
}
DEFAULTS = dict(ratio=0.75, k=11, cls="bce", gamma=1.5, alpha=0.25, down="haar", refine=1, gate=1)
# Control experiments for AW-B (GC10-DET), all built on `awf-awb`:
#   awf-awb@down=spd            same block with space-to-depth (pixel-unshuffle) instead of the Haar transform
#   awf-awb@gate=0              AW-B without the sub-band gate
#   awf-awb@refine=0            AW-B without the depthwise refinement
#   awf-awb@refine=0,gate=0     Haar transform + 1x1 projection only (HWD-style downsampling)


def parse_variant(cfg):
    """'awf-yolo@ratio=0.8,cls=focal' or a dict (from a checkpoint) -> full spec dict."""
    if isinstance(cfg, dict):
        return dict(cfg.get("awf", cfg))
    name, _, opts = str(cfg).partition("@")
    name = name.strip()
    if name not in VARIANTS:
        raise ValueError(f"Unknown AWF variant '{name}'. Choose from {list(VARIANTS)}")
    spec = {"name": name, **VARIANTS[name], **DEFAULTS}
    for kv in filter(None, opts.split(",")):
        k, v = kv.split("=")
        spec[k.strip()] = type(DEFAULTS.get(k.strip(), ""))(v) if k.strip() in DEFAULTS else v
    spec["tag"] = str(cfg)
    return spec


def build_layers(spec, nc, ch=3):
    awb, mrff, caa = spec["awb"], spec["mrff"], spec["caa"]
    layers = []

    def add(m, f=-1):
        m.f, m.i, m.type = f, len(layers), m.__class__.__name__
        m.np = sum(p.numel() for p in m.parameters())
        layers.append(m)
        return m.i

    def down(c1, c2):
        if not awb:
            return Conv(c1, c2, 3, 2)
        return AWB(c1, c2, transform=str(spec.get("down", "haar")), refine=bool(int(spec.get("refine", 1))),
                   gate=bool(int(spec.get("gate", 1))))

    # Backbone (YOLO11n channels: 16-32-64-128-256)
    add(down(ch, 16))                                   # 0  P1/2
    add(down(16, 32))                                   # 1  P2/4
    add(C3k2(32, 64, 1, False, 0.25))                   # 2
    add(down(64, 64))                                   # 3  P3/8
    p3b = add(C3k2(64, 128, 1, False, 0.25))            # 4
    add(down(128, 128))                                 # 5  P4/16
    p4b = add(C3k2(128, 128, 1, True))                  # 6
    add(down(128, 256))                                 # 7  P5/32
    add(C3k2(256, 256, 1, True))                        # 8
    add(SPPF(256, 256, 5))                              # 9
    p5 = add(C2PSA_CAA(256, 256, 1, k=int(spec["k"])) if caa else C2PSA(256, 256, 1))  # 10

    # Top-down path
    if mrff:
        p4t = add(MRFF(256, 128, 128, 1), [-1, p4b])
        p3o = add(MRFF(128, 128, 64, 1), [-1, p3b])
    else:
        add(nn.Upsample(None, 2, "nearest"))
        add(Concat(1), [-1, p4b])
        p4t = add(C3k2(384, 128, 1, False))
        add(nn.Upsample(None, 2, "nearest"))
        add(Concat(1), [-1, p3b])
        p3o = add(C3k2(256, 64, 1, False))

    # Bottom-up path and head (identical in all variants)
    add(Conv(64, 64, 3, 2))
    add(Concat(1), [-1, p4t])
    p4o = add(C3k2(192, 128, 1, False))
    add(Conv(128, 128, 3, 2))
    add(Concat(1), [-1, p5])
    p5o = add(C3k2(384, 256, 1, True))
    add(Detect(nc=nc, ch=(64, 128, 256)), [p3o, p4o, p5o])

    save = sorted({j for m in layers for j in ([m.f] if isinstance(m.f, int) else m.f) if j != -1})
    return layers, save


class AWFDetectionModel(DetectionModel):
    """Ultralytics-compatible AWF-YOLO. Works with DetectionTrainer, val(), predict() and export()."""

    def __init__(self, cfg="awf-yolo", ch=3, nc=10, verbose=True, class_weights=None):
        nn.Module.__init__(self)
        spec = parse_variant(cfg)
        if class_weights is not None:
            spec["class_weights"] = [float(w) for w in class_weights]
        self.awf = spec
        self.yaml = {"awf": spec, "nc": nc, "channels": ch, "ch": ch, "yaml_file": f"{spec['name']}.awf"}
        layers, self.save = build_layers(spec, nc, ch)
        self.model = nn.Sequential(*layers)
        self.names = {i: str(i) for i in range(nc)}
        self.inplace = True
        try:
            self.end2end = getattr(self.model[-1], "end2end", False)
        except AttributeError:
            pass

        detect = self.model[-1]
        detect.inplace = True
        s = 256
        self.eval()
        with torch.no_grad():
            feats = self._detect_inputs(torch.zeros(1, ch, s, s))
        self.train()
        detect.stride = torch.tensor([s / f.shape[-2] for f in feats])
        self.stride = detect.stride
        detect.bias_init()
        initialize_weights(self)
        if verbose:
            n = sum(p.numel() for p in self.parameters())
            print(f"AWF model '{spec.get('tag', spec['name'])}': {len(self.model)} layers, {n:,} parameters")

    def _detect_inputs(self, x):
        y = []
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return [x if j == -1 else y[j] for j in self.model[-1].f]

    def init_criterion(self):
        if self.awf.get("inner"):
            enable_inner_siou(float(self.awf.get("ratio", 0.75)))
        criterion = super().init_criterion()
        mode = self.awf.get("cls", "bce")
        if mode != "bce":
            target = criterion if hasattr(criterion, "bce") else getattr(criterion, "one2many", None)
            if target is None or not hasattr(target, "bce"):
                raise RuntimeError("Could not find the classification BCE in this Ultralytics loss.")
            if mode == "focal":
                target.bce = FocalBCE(float(self.awf.get("gamma", 1.5)), float(self.awf.get("alpha", 0.25)))
            elif mode == "weighted":
                target.bce = WeightedBCE(self.awf["class_weights"])
            else:
                raise ValueError(f"Unknown cls loss '{mode}'")
        return criterion
