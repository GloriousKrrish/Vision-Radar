import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

class AccidentClassifierService:
    """
    Secondary Candidate-Frame Verifier Service.
    Loads data/models/accident_classifier/accident_classifier.onnx (or PyTorch .pt fallback) lazily
    only when candidate incident frames are extracted.
    
    Section 4 & 5 & 6 Compliant:
    - Input: 224x224 RGB image, ImageNet mean/std normalization
    - Output: [1, 2] logits -> Softmax probabilities (0 = NORMAL_TRAFFIC, 1 = ACCIDENT_CANDIDATE)
    - Multi-frame temporal aggregation (policy_version: accident_temporal_v1.0.0)
    """
    
    def __init__(self, model_path: str = "data/models/accident_classifier/accident_classifier.onnx"):
        self.model_path = model_path
        self.onnx_session = None
        self.torch_model = None
        self.is_loaded = False
        
        # Telemetry counters
        self.invocations = 0
        self.candidate_frames_classified = 0
        self.normal_predictions = 0
        self.accident_predictions = 0
        self.total_preprocessing_ms = 0.0
        self.total_inference_ms = 0.0
        self.total_postprocessing_ms = 0.0
        
        # Mean & Std for ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _ensure_loaded(self):
        if self.is_loaded:
            return
        
        # Try OpenCV DNN ONNX loading first
        onnx_file = self.model_path
        if not os.path.exists(onnx_file):
            onnx_file = "data/models/accident_classifier/accident_classifier.onnx"
            
        if os.path.exists(onnx_file):
            try:
                self.net = cv2.dnn.readNetFromONNX(onnx_file)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.is_loaded = True
                print(f"[ACCIDENT CLASSIFIER] Loaded ONNX model from {onnx_file} via OpenCV DNN.")
                return
            except Exception as e:
                print(f"[ACCIDENT CLASSIFIER] OpenCV DNN ONNX load failed: {e}")
                
        # Try PyTorch fallback if .pt file exists
        pt_file = "data/models/accident_classifier/accident_classifier.pt"
        if os.path.exists(pt_file):
            try:
                import torch
                import torch.nn as nn
                
                class LightweightAccidentClassifier(nn.Module):
                    def __init__(self):
                        super(LightweightAccidentClassifier, self).__init__()
                        self.features = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d((1, 1))
                        )
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, 2)
                        )
                    def forward(self, x):
                        x = self.features(x)
                        x = torch.flatten(x, 1)
                        return self.classifier(x)
                        
                model = LightweightAccidentClassifier()
                model.load_state_dict(torch.load(pt_file, map_location="cpu"))
                model.eval()
                self.torch_model = model
                self.is_loaded = True
                print(f"[ACCIDENT CLASSIFIER] Loaded PyTorch fallback model from {pt_file}.")
                return
            except Exception as e:
                print(f"[ACCIDENT CLASSIFIER] PyTorch load failed: {e}")
                
        raise RuntimeError(f"Accident classifier model missing or unreadable at {self.model_path}")

    def preprocess_image(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Exact preprocessing contract:
        - Resize BGR image to 224x224
        - Convert BGR -> RGB
        - Normalize float32 range [0..1]
        - Apply ImageNet mean/std normalization
        - Return [1, 3, 224, 224] NCHW blob
        """
        t0 = time.perf_counter()
        resized = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        # Transpose HWC (224, 224, 3) -> CHW (3, 224, 224) -> NCHW (1, 3, 224, 224)
        chw = np.transpose(normalized, (2, 0, 1))
        nchw = np.expand_dims(chw, axis=0)
        t_ms = (time.perf_counter() - t0) * 1000.0
        return nchw, t_ms

    def predict_frame(self, frame_bgr: np.ndarray, frame_index: int = 0) -> Dict[str, Any]:
        """
        Classifies a single candidate image frame.
        """
        self._ensure_loaded()
        self.invocations += 1
        self.candidate_frames_classified += 1

        blob, prep_ms = self.preprocess_image(frame_bgr)
        self.total_preprocessing_ms += prep_ms

        t0_inf = time.perf_counter()
        if hasattr(self, 'net') and self.net is not None:
            self.net.setInput(blob)
            logits = self.net.forward()[0]
        elif self.torch_model is not None:
            import torch
            with torch.no_grad():
                tensor_in = torch.from_numpy(blob)
                logits = self.torch_model(tensor_in).numpy()[0]
        else:
            logits = np.array([1.0, 0.0], dtype=np.float32)
        inf_ms = (time.perf_counter() - t0_inf) * 1000.0
        self.total_inference_ms += inf_ms

        t0_post = time.perf_counter()
        # Softmax computation
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        normal_prob = float(probs[0])
        accident_prob = float(probs[1])

        pred_class_idx = int(np.argmax(probs))
        pred_class = "ACCIDENT_CANDIDATE" if pred_class_idx == 1 else "NORMAL_TRAFFIC"

        if pred_class_idx == 1:
            self.accident_predictions += 1
        else:
            self.normal_predictions += 1

        post_ms = (time.perf_counter() - t0_post) * 1000.0
        self.total_postprocessing_ms += post_ms

        return {
            "frame_index": frame_index,
            "predicted_class": pred_class,
            "class_index": pred_class_idx,
            "accident_probability": round(accident_prob, 4),
            "normal_probability": round(normal_prob, 4),
            "preprocessing_ms": round(prep_ms, 2),
            "inference_ms": round(inf_ms, 2),
            "postprocessing_ms": round(post_ms, 2),
            "model_file": os.path.basename(self.model_path)
        }

    def aggregate_candidate_frames(
        self,
        candidate_frames: List[Tuple[int, np.ndarray]], # [(frame_index, frame_bgr)]
        policy_version: str = "accident_temporal_v1.0.0"
    ) -> Dict[str, Any]:
        """
        Classifies candidate frames (e.g. before, event, after) and performs temporal aggregation.
        """
        if not candidate_frames:
            return {
                "candidate_score": 0.0,
                "frame_predictions": [],
                "aggregation_method": "MEAN_PROBABILITY_THRESHOLD",
                "policy_version": policy_version,
                "final_class": "NORMAL_TRAFFIC",
                "is_candidate": False
            }

        preds = [self.predict_frame(f[1], frame_index=f[0]) for f in candidate_frames]
        acc_probs = [p["accident_probability"] for p in preds]
        mean_score = float(np.mean(acc_probs)) if acc_probs else 0.0
        max_score = float(np.max(acc_probs)) if acc_probs else 0.0

        # Temporal policy: Mean probability >= 0.50 or any candidate frame > 0.70
        is_candidate = (mean_score >= 0.50) or (max_score >= 0.70)
        final_class = "ACCIDENT_CANDIDATE" if is_candidate else "NORMAL_TRAFFIC"

        return {
            "candidate_score": round(mean_score, 4),
            "max_frame_score": round(max_score, 4),
            "frame_predictions": preds,
            "aggregation_method": "MEAN_AND_PEAK_PERSISTENCE",
            "policy_version": policy_version,
            "final_class": final_class,
            "is_candidate": is_candidate
        }

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "accident_classifier_loaded": self.is_loaded,
            "accident_classifier_invocations": self.invocations,
            "candidate_frames_classified": self.candidate_frames_classified,
            "normal_predictions": self.normal_predictions,
            "accident_predictions": self.accident_predictions,
            "total_preprocessing_ms": round(self.total_preprocessing_ms, 2),
            "total_inference_ms": round(self.total_inference_ms, 2),
            "total_postprocessing_ms": round(self.total_postprocessing_ms, 2),
            "total_classifier_ms": round(self.total_preprocessing_ms + self.total_inference_ms + self.total_postprocessing_ms, 2),
            "policy_version": "accident_temporal_v1.0.0"
        }
