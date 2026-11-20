"""Optional edge-device benchmark (not used for the results in the paper).
On Colab:   python edge_benchmark.py export --weights runs/gc10_awf-yolo/weights/best.pt
On device:  python edge_benchmark.py run --onnx best.onnx --images test_images/ --n 300
            (Jetson: run `tegrastats --interval 500 --logfile power.log` in parallel for power)"""
import argparse, glob, os, time

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
e = sub.add_parser("export"); e.add_argument("--weights", required=True); e.add_argument("--imgsz", type=int, default=640)
r = sub.add_parser("run"); r.add_argument("--onnx", required=True); r.add_argument("--images", required=True)
r.add_argument("--n", type=int, default=300); r.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()
if a.cmd == "export":
    import _path  # noqa: F401
    import awf  # noqa: F401
    from ultralytics import YOLO
    p = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True)
    print("exported", p, f"{os.path.getsize(p) / 1e6:.2f} MB")
else:
    import cv2, numpy as np, onnxruntime as ort, psutil
    proc = psutil.Process(); rss0 = proc.memory_info().rss
    t0 = time.perf_counter(); sess = ort.InferenceSession(a.onnx, providers=ort.get_available_providers())
    load_ms = (time.perf_counter() - t0) * 1000
    name = sess.get_inputs()[0].name
    files = sorted(glob.glob(os.path.join(a.images, "*")))[:a.n]
    prep = lambda f: np.ascontiguousarray(cv2.resize(cv2.imread(f), (a.imgsz, a.imgsz))[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    x = prep(files[0])
    for _ in range(20):
        sess.run(None, {name: x})
    lat, peak = [], 0
    for f in files:
        x = prep(f); t = time.perf_counter(); sess.run(None, {name: x}); lat.append((time.perf_counter() - t) * 1000)
        peak = max(peak, proc.memory_info().rss)
    lat = np.array(lat)
    print(f"providers {sess.get_providers()} | model file {os.path.getsize(a.onnx) / 1e6:.2f} MB | load {load_ms:.0f} ms")
    print(f"latency mean {lat.mean():.1f} ms, median {np.median(lat):.1f}, p95 {np.percentile(lat, 95):.1f} -> FPS {1000 / lat.mean():.1f}")
    print(f"peak process memory {peak / 1e6:.0f} MB (+{(peak - rss0) / 1e6:.0f} MB)")
