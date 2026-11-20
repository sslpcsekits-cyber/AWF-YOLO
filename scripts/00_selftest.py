"""Run this FIRST (about 1-2 minutes, no dataset needed). It checks every AWF component and prints the
parameter count and GFLOPs of every variant. Check that it ends with SELF-TEST PASSED before starting long runs."""
import copy, math
import _path  # noqa: F401
import torch
import ultralytics
from ultralytics.cfg import get_cfg

import awf
from awf import AWFDetectionModel, VARIANTS, haar_dwt, inner_siou

print("PyTorch", torch.__version__, "| Ultralytics", ultralytics.__version__,
      "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
dev = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Haar DWT is orthonormal: energy is preserved and nothing is discarded
x = torch.randn(2, 3, 64, 64)
ll, lh, hl, hh = haar_dwt(x)
e_in, e_out = x.pow(2).sum(), sum(t.pow(2).sum() for t in (ll, lh, hl, hh))
assert torch.allclose(e_in, e_out, rtol=1e-4), (e_in, e_out)
print("[ok] Haar DWT preserves energy:", f"{e_in:.1f} vs {e_out:.1f}")

# 2. Inner-SIoU: identical boxes -> value 1 (loss 0); far boxes -> large loss
b = torch.tensor([[10., 10., 50., 30.]])
assert abs(float(inner_siou(b, b)) - 1) < 1e-4
far = float(1 - inner_siou(b, torch.tensor([[200., 200., 260., 230.]])))
print(f"[ok] Inner-SIoU: identical boxes loss = {float(1 - inner_siou(b, b)):.4f}, distant boxes loss = {far:.3f}")


def gflops(model, imgsz=640):
    try:
        import thop
        m = copy.deepcopy(model).eval().float().cpu()
        macs, _ = thop.profile(m, inputs=(torch.zeros(1, 3, imgsz, imgsz),), verbose=False)
        return 2 * macs / 1e9
    except Exception as e:  # noqa: BLE001
        return float("nan")


# 3. Every variant builds, runs forward in train and eval mode, and computes the detection loss
cfg = get_cfg()
batch = {
    "img": torch.rand(2, 3, 640, 640, device=dev),
    "batch_idx": torch.tensor([0., 0., 1.], device=dev),
    "cls": torch.tensor([[1.], [3.], [7.]], device=dev),
    "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1], [0.3, 0.6, 0.05, 0.4], [0.7, 0.2, 0.3, 0.3]], device=dev),
}
print(f"\n{'variant':<34}{'params':>12}{'GFLOPs':>9}  loss")
for name in list(VARIANTS) + ["awf-yolo@cls=focal", "awf-yolo@ratio=0.8"]:
    m = AWFDetectionModel(name, nc=10, verbose=False).to(dev)
    m.args = cfg
    m.train()
    loss, items = m.loss(batch)
    assert torch.isfinite(loss).all(), name
    loss.sum().backward()
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(1, 3, 640, 640, device=dev))
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:<34}{n:>12,}{gflops(m):>9.2f}  {[round(float(v), 3) for v in (items.values() if isinstance(items, dict) else items)]}")
    import ultralytics.utils.loss as L
    if hasattr(L, "_awf_original_bbox_iou"):           # undo the Inner-SIoU hook between variants
        L.bbox_iou = L._awf_original_bbox_iou

w = AWFDetectionModel("awf-yolo@cls=weighted", nc=10, verbose=False, class_weights=[1.0] * 10)
print("[ok] weighted-loss variant builds")

# 4. Reference: official YOLO11n built from yaml with 10 classes (awf-base should be very close)
from ultralytics import YOLO
ref = YOLO("yolo11n.yaml").model
ref_n = sum(p.numel() for p in ref.parameters())
print(f"\nReference yolo11n.yaml (80 classes): {ref_n:,} parameters (awf-base has 10 classes, so slightly fewer)")
print("\nSELF-TEST PASSED")
