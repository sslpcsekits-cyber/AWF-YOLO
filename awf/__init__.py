"""AWF-YOLO: adaptive wavelet-based frequency-alignment YOLO (clean re-implementation).

Importing this package registers nothing globally; it only makes the custom model class importable,
which Ultralytics needs when it re-loads an AWF-YOLO checkpoint (best.pt / last.pt).
"""
from .modules import haar_dwt, AWB, EUCB, MRFF, CAA, C2PSA_CAA
from .losses import inner_siou, enable_inner_siou, FocalBCE, WeightedBCE
from .model import AWFDetectionModel, VARIANTS, parse_variant
from .trainer import AWFTrainer, PROTOCOL, train

__all__ = ["haar_dwt", "AWB", "EUCB", "MRFF", "CAA", "C2PSA_CAA", "inner_siou", "enable_inner_siou",
           "FocalBCE", "WeightedBCE", "AWFDetectionModel", "VARIANTS", "parse_variant",
           "AWFTrainer", "PROTOCOL", "train"]
