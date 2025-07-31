#!/usr/bin/env python3
"""
shim_retinaface.py

A thin wrapper that exposes a RetinaFace-like API:

    model = RetinaFace.build_model()
    dets  = RetinaFace.detect_faces(frame_bgr, model, threshold=0.6)

Primary backend: InsightFace (ONNXRuntime-GPU) → fast & robust on CUDA.
Fallbacks: retina-face (TF) or retinaface-pytorch (CPU) if InsightFace/GPU is missing.

Returns a dict:
{
  "face_1": {"score": float, "facial_area": [x1, y1, x2, y2]},
  "face_2": {...},
  ...
}
"""

from __future__ import annotations
import os
import numpy as np

# Quiet ONNX Runtime logs (0=VERBOSE,1=INFO,2=WARN,3=ERROR,4=FATAL)
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

# ---- Try InsightFace (preferred) -------------------------------------------
INSIGHT_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis  # uses onnxruntime/onnxruntime-gpu
    import onnxruntime as ort
    INSIGHT_AVAILABLE = True
except Exception:
    pass

# ---- Fallbacks: TF & PyTorch implementations -------------------------------
TF_AVAILABLE = False
PT_AVAILABLE = False
try:
    # pip install retina-face (TensorFlow/Keras)
    from retinaface import RetinaFace as _TF_RetinaFace
    TF_AVAILABLE = True
except Exception:
    pass

try:
    # pip install retinaface-pytorch
    from retinaface.pre_trained_models import get_model as _rf_get_model
    PT_AVAILABLE = True
except Exception:
    pass


class RetinaFace:
    _fa = None  # InsightFace FaceAnalysis instance

    @staticmethod
    def _providers_ok() -> bool:
        """Return True if ORT reports a GPU (CUDA or TensorRT) provider."""
        try:
            dev = ort.get_device()
            provs = set(ort.get_available_providers())
            return (dev == "GPU") and (
                "CUDAExecutionProvider" in provs or "TensorrtExecutionProvider" in provs
            )
        except Exception:
            return False

    @staticmethod
    def build_model():
        """
        Build and return a detector model object.
        Prefers InsightFace with CUDA; falls back to CPU implementations.
        """
        # 1) InsightFace (GPU) path
        if INSIGHT_AVAILABLE and RetinaFace._providers_ok():
            # Force CUDA; if it fails, we will fall back to CPU cleanly.
            providers = ["CUDAExecutionProvider"]
            try:
                fa = FaceAnalysis(name="buffalo_l", providers=providers)
                # Larger det_size → better small-face recall (slower). 960 is a good trade-off.
                fa.prepare(ctx_id=0, det_size=(960, 960))
                # Debug: print actual providers
                try:
                    sess = getattr(fa, "det_model", None) and fa.det_model.session
                    if sess:
                        print("[InsightFace] providers:", sess.get_providers())
                except Exception:
                    pass
                RetinaFace._fa = fa
                return fa
            except Exception as e:
                print(f"[InsightFace] CUDA provider failed ({e}); falling back to CPU.")

        # 2) InsightFace (CPU) fallback
        if INSIGHT_AVAILABLE:
            try:
                fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                fa.prepare(ctx_id=-1, det_size=(960, 960))
                RetinaFace._fa = fa
                return fa
            except Exception:
                pass

        # 3) TensorFlow/Keras retina-face fallback
        if TF_AVAILABLE:
            try:
                return _TF_RetinaFace.build_model()
            except Exception:
                pass

        # 4) PyTorch retinaface-pytorch fallback (kept CPU-safe to avoid device mismatch)
        if PT_AVAILABLE:
            m = _rf_get_model("resnet50_2020-07-20", max_size=2048)
            m.eval()
            return m

        raise ImportError(
            "No face detector available. Install one of:\n"
            "  pip install insightface onnxruntime-gpu\n"
            "  pip install retina-face\n"
            "  pip install retinaface-pytorch"
        )

    @staticmethod
    def detect_faces(frame_bgr: np.ndarray, model, threshold: float = 0.6):
        """
        Detect faces in a BGR frame. Returns dict keyed by 'face_#' with bbox + score.
        Default threshold 0.6 is appropriate for InsightFace; adjust via caller's --conf.
        """
        thr = float(threshold or 0.6)

        # 1) InsightFace route
        if INSIGHT_AVAILABLE and isinstance(model, FaceAnalysis):
            # InsightFace expects RGB input
            rgb = frame_bgr[..., ::-1]
            faces = model.get(rgb)
            out, k = {}, 1
            h, w = frame_bgr.shape[:2]
            for f in faces:
                score = float(getattr(f, "det_score", 1.0))
                if score < thr:
                    continue
                # f.bbox is [x1, y1, x2, y2] in float
                x1, y1, x2, y2 = map(int, np.clip(f.bbox, [0, 0, 0, 0], [w, h, w, h]))
                if x2 <= x1 or y2 <= y1:
                    continue
                out[f"face_{k}"] = {"score": score, "facial_area": [x1, y1, x2, y2]}
                k += 1
            return out

        # 2) TensorFlow/Keras retina-face
        if TF_AVAILABLE and hasattr(model, "predict"):
            return _TF_RetinaFace.detect_faces(frame_bgr, model=model, threshold=thr)

        # 3) PyTorch retinaface-pytorch (CPU-only here)
        if PT_AVAILABLE and hasattr(model, "predict_jsons"):
            anns = model.predict_jsons(frame_bgr)
            out, k = {}, 1
            if anns:
                h, w = frame_bgr.shape[:2]
                for det in anns:
                    score = float(det.get("score", 0.0))
                    if score < thr:
                        continue
                    bbox = det.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w));     y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    out[f"face_{k}"] = {"score": score, "facial_area": [x1, y1, x2, y2]}
                    k += 1
            return out

        # No backend available
        return {}
