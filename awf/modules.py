"""Building blocks of AWF-YOLO.

AWB        Adaptive Wavelet Bottleneck: 2x downsampling by a Haar discrete wavelet transform (DWT),
           which keeps all four sub-bands (LL, LH, HL, HH) instead of discarding high frequencies
           like a strided convolution; each sub-band is refined by a depthwise 3x3 convolution and
           re-weighted by an input-adaptive gate before a 1x1 fusion convolution.
EUCB       Efficient up-sampling and convolution block: 2x nearest upsampling -> depthwise 3x3 -> 1x1.
MRFF       Multi-Resolution Frequency Fusion: fuses an EUCB-upsampled deep map with a lateral map whose
           high-frequency residual (x - lowpass(x)) is re-emphasised by a learnable per-channel gain.
CAA        Context Anchor Attention with asymmetric strip convolutions (1xk then kx1).
C2PSA_CAA  Ultralytics C2PSA block followed by CAA on the attention branch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C2PSA, C3k2, Conv


def haar_dwt(x):
    """One-level 2D Haar DWT. Returns (LL, LH, HL, HH), each at half resolution. Orthonormal (factor 1/2)."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (-a - b + c + d) * 0.5
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


class AWB(nn.Module):
    """Adaptive Wavelet Bottleneck: replaces a stride-2 convolution (c1 -> c2, H x W -> H/2 x W/2)."""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        c4 = 4 * c1
        self.c1 = c1
        self.refine = nn.Sequential(nn.Conv2d(c4, c4, k, 1, k // 2, groups=c4, bias=False),
                                    nn.BatchNorm2d(c4), nn.SiLU())
        self.gate = nn.Conv2d(c4, 4, 1, groups=4, bias=True)  # one adaptive weight per sub-band
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)                          # 2*sigmoid(0) = 1 -> identity at start
        self.fuse = Conv(c4, c2, 1, 1)

    def forward(self, x):
        s = torch.cat(haar_dwt(x), 1)
        s = s + self.refine(s)
        w = 2 * torch.sigmoid(self.gate(F.adaptive_avg_pool2d(s, 1)))   # (B, 4, 1, 1)
        s = s * w.repeat_interleave(self.c1, dim=1)
        return self.fuse(s)


class EUCB(nn.Module):
    """Efficient up-sampling and convolution block (2x)."""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dw = Conv(c1, c1, k, 1, g=c1)
        self.pw = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(self.up(x)))


class MRFF(nn.Module):
    """Multi-Resolution Frequency Fusion. Input: [deep (stride 2s), lateral (stride s)]."""

    def __init__(self, c_deep, c_lat, c2, n=1):
        super().__init__()
        self.up = EUCB(c_deep, c_lat)
        self.hf_gain = nn.Parameter(torch.zeros(1, c_lat, 1, 1))   # starts as plain fusion
        self.fuse = C3k2(2 * c_lat, c2, n, False)

    def forward(self, xs):
        deep, lat = xs
        low = F.interpolate(F.avg_pool2d(lat, 2, ceil_mode=True), size=lat.shape[-2:], mode="nearest")
        lat = lat + self.hf_gain * (lat - low)
        up = self.up(deep)
        if up.shape[-2:] != lat.shape[-2:]:
            up = F.interpolate(up, size=lat.shape[-2:], mode="nearest")
        return self.fuse(torch.cat((up, lat), 1))


class CAA(nn.Module):
    """Context Anchor Attention: pooled context -> 1x1 -> 1xk -> kx1 (depthwise) -> 1x1 -> sigmoid."""

    def __init__(self, c, k=11):
        super().__init__()
        self.pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(c, c, 1, 1)
        self.h = nn.Conv2d(c, c, (1, k), 1, (0, k // 2), groups=c)
        self.v = nn.Conv2d(c, c, (k, 1), 1, (k // 2, 0), groups=c)
        self.conv2 = Conv(c, c, 1, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv2(self.v(self.h(self.conv1(self.pool(x))))))


class C2PSA_CAA(C2PSA):
    """C2PSA with Context Anchor Attention (strip convolutions) on the attention branch."""

    def __init__(self, c1, c2, n=1, e=0.5, k=11):
        super().__init__(c1, c2, n, e)
        self.caa = CAA(self.c, k)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        b = b * self.caa(b)
        return self.cv2(torch.cat((a, b), 1))
