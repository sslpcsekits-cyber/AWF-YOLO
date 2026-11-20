"""Faster, numerically equivalent inference path for AWB (use for latency measurements and deployment).

The Haar DWT in modules.haar_dwt is computed with four strided slices and twelve element-wise operations
followed by a concatenation. Here it is one grouped 2x2 stride-2 convolution with fixed Haar kernels, and the
sub-band gate is applied by broadcasting instead of repeat_interleave. Outputs are identical up to floating-point
rounding; trained checkpoints are unchanged.

    from awf.fast import fast_patch, check_equivalence
    n = fast_patch(model.model)          # model = YOLO('best.pt'); returns number of AWB modules patched
"""
import types
import torch
import torch.nn.functional as F
from .modules import AWB

# rows: LL, LH, HL, HH; columns: positions (0,0) (0,1) (1,0) (1,1) of each 2x2 block, as in modules.haar_dwt
_HAAR = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                      [-0.5, -0.5, 0.5, 0.5],
                      [-0.5, 0.5, -0.5, 0.5],
                      [0.5, -0.5, -0.5, 0.5]]).view(4, 1, 2, 2)


def _fast_forward(self, x):
    B, C, H, W = x.shape
    if H % 2 or W % 2:
        return AWB.forward(self, x)                     # odd sizes: original path (replicate padding)
    key = (C, x.dtype, x.device)
    cache = self.__dict__.setdefault("_haar_w", {})
    if key not in cache:
        cache[key] = _HAAR.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
    h, w = H // 2, W // 2
    s = F.conv2d(x, cache[key], stride=2, groups=C)     # (B, 4C) channel-major: c0[LL,LH,HL,HH], c1[...]
    s = s.view(B, C, 4, h, w).transpose(1, 2).reshape(B, 4 * C, h, w)   # band-major, as torch.cat(haar_dwt(x), 1)
    s = s + self.refine(s)
    g = 2 * torch.sigmoid(self.gate(F.adaptive_avg_pool2d(s, 1)))      # (B, 4, 1, 1)
    s = (s.view(B, 4, C, h, w) * g.view(B, 4, 1, 1, 1)).view(B, 4 * C, h, w)
    return self.fuse(s)


def fast_patch(model):
    """Replace AWB.forward by the fast path in every AWB module of `model` (an nn.Module)."""
    n = 0
    for m in model.modules():
        if isinstance(m, AWB):
            m.forward = types.MethodType(_fast_forward, m)
            n += 1
    return n


@torch.no_grad()
def check_equivalence(model_factory, size=640, device="cpu"):
    """model_factory() must return a fresh nn.Module. Returns the max abs difference of the raw outputs."""
    ref = model_factory().eval().float().to(device)
    fast = model_factory().eval().float().to(device)
    fast_patch(fast)
    x = torch.rand(2, 3, size, size, device=device)
    a, b = ref(x), fast(x)
    a = a[0] if isinstance(a, (list, tuple)) else a
    b = b[0] if isinstance(b, (list, tuple)) else b
    return (a - b).abs().max().item()
