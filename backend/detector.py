import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import mediapipe as mp
import torch
import torchvision.transforms as transforms

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


@dataclass
class FrameAnalysis:
    frame_idx: int
    has_face: bool
    ear: float
    mar: float
    face_confidence: float
    texture_score: float
    artifact_score: float
    dl_score: float = 0.5   # ✅ NEW
    landmarks: Optional[np.ndarray] = None


@dataclass
class VideoResult:
    is_fake: bool
    confidence: float
    frame_results: List[FrameAnalysis]
    summary: Dict
    message: str


class DeepfakeDetector:
    def __init__(self, model_weights_path: Optional[str] = None):

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 13, 14, 78, 308]

        # ✅ PyTorch setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dl_model = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if model_weights_path and os.path.exists(model_weights_path):
            self._load_dl_model(model_weights_path)

    def _load_dl_model(self, path):
        try:
            import torchvision.models as models
            import torch.nn as nn

            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()

            self.dl_model = model
            print("✅ PyTorch model loaded")

        except Exception as e:
            print(f"DL model load failed: {e}")
            self.dl_model = None

    def _predict_dl(self, frame: np.ndarray) -> float:
        if self.dl_model is None:
            return 0.5

        try:
            img = self.transform(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.dl_model(img)

            return float(pred.item())

        except Exception:
            return 0.5

    def _calculate_ear(self, landmarks, eye_indices):
        pts = landmarks[eye_indices]
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0

    def _calculate_mar(self, landmarks):
        mouth = landmarks[self.MOUTH]
        A = np.linalg.norm(mouth[1] - mouth[0])
        B = np.linalg.norm(mouth[3] - mouth[2])
        return B / A if A != 0 else 0.0

    def _analyze_texture(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        return min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)

    def _detect_artifacts(self, frame, bbox):
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            return 0.5

        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)

        blur = cv2.Laplacian(l, cv2.CV_64F).var()
        return 1.0 - min(blur / 300.0, 1.0)

    def _get_face_bbox(self, landmarks, shape):
        h, w = shape[:2]
        xs, ys = landmarks[:, 0], landmarks[:, 1]

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        pad = 20
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad) - max(0, x1 - pad),
            min(h, y2 + pad) - max(0, y1 - pad)
        )

    def analyze_frame(self, frame, idx):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        det = self.face_detection.process(rgb)

        if not det.detections:
            return FrameAnalysis(idx, False, 0, 0, 0, 0, 0, 0.5)

        mesh = self.face_mesh.process(rgb)

        if not mesh.multi_face_landmarks:
            return FrameAnalysis(idx, True, 0, 0, det.detections[0].score[0], 0, 0, 0.5)

        lm = mesh.multi_face_landmarks[0]
        landmarks = np.array([[p.x * w, p.y * h] for p in lm.landmark])

        ear = (self._calculate_ear(landmarks, self.LEFT_EYE) +
               self._calculate_ear(landmarks, self.RIGHT_EYE)) / 2

        mar = self._calculate_mar(landmarks)

        bbox = self._get_face_bbox(landmarks, frame.shape)
        x, y, fw, fh = bbox

        if fw > 0 and fh > 0:
            roi = frame[y:y+fh, x:x+fw]
            texture = self._analyze_texture(roi)
            artifact = self._detect_artifacts(frame, bbox)
        else:
            texture, artifact = 0.5, 0.5

        # ✅ DL prediction
        dl_score = self._predict_dl(frame)

        return FrameAnalysis(
            idx, True, ear, mar,
            det.detections[0].score[0],
            texture, artifact,
            dl_score,
            landmarks
        )

    def _heuristic_classification(self, frames):
        valid = [f for f in frames if f.has_face]

        if len(valid) < 5:
            # ✅ Convert to native Python types
            return False, 0.0, {}

        ear_std = np.std([f.ear for f in valid])
        texture_mean = np.mean([f.texture_score for f in valid])
        artifact_mean = np.mean([f.artifact_score for f in valid])
        dl_mean = np.mean([f.dl_score for f in valid])

        heuristic_score = (
            (0.8 if ear_std < 0.015 else 0.3) * 0.3 +
            (0.7 if texture_mean < 0.2 else 0.3) * 0.3 +
            artifact_mean * 0.4
        )

        final_score = (heuristic_score * 0.6) + (dl_mean * 0.4)
        confidence = min(float(final_score), 1.0)  # ✅ Convert to float

        # ✅ Convert numpy.bool_ to Python bool
        is_fake = bool(confidence > 0.5)

        return is_fake, confidence, {
            "dl_score": float(dl_mean),
            "heuristic_score": float(heuristic_score)
        }

    def analyze_video(self, video_path, sample_rate=5):
        if not os.path.exists(video_path):
            return VideoResult(False, 0.0, [], {}, "Video not found")  # ✅ 0.0 not 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoResult(False, 0.0, [], {}, "Cannot open video")  # ✅ 0.0 not 0

        results = []
        i = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if i % sample_rate == 0:
                results.append(self.analyze_frame(frame, i))

            i += 1

        cap.release()

        if not results:
            return VideoResult(False, 0.0, [], {}, "No frames")  # ✅ 0.0 not 0

        is_fake, conf, details = self._heuristic_classification(results)

        return VideoResult(
            bool(is_fake),           # ✅ Ensure native bool
            float(conf),             # ✅ Ensure native float
            results,
            details,
            "LIKELY FAKE" if conf > 0.7 else "LIKELY REAL"
        )