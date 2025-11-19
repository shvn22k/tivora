# src/core/face_parsing.py
import os
import numpy as np
import cv2
import onnxruntime as ort

class FaceParser:
    def __init__(self, model_path="src/models/face_parse_bisenet.onnx", input_size=(512,512), provider=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.input_size = input_size
        providers = [provider] if provider else None
        self.session = ort.InferenceSession(model_path, providers=providers) if providers else ort.InferenceSession(model_path)
        # try to find the single useful output index
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        h, w = self.input_size
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
        # shape (1,3,H,W)
        img = np.transpose(img, (2,0,1))[None, ...].astype(np.float32)
        return img

    def postprocess(self, logits, frame_shape):
        # logits expected shape (1, C, H, W) -> class map HxW
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim == 4:
            probs = logits[0]  # (C,H,W)
        elif logits.ndim == 3:
            probs = logits
        else:
            # unknown shape -> return zeros
            H, W = frame_shape[:2]
            return np.zeros((H,W), dtype=np.uint8)
        cls = np.argmax(probs, axis=0).astype(np.uint8)
        cls = cv2.resize(cls, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        return cls

    def get_segmentation(self, frame):
        inp = self.preprocess(frame)
        try:
            outs = self.session.run(None, {self.session.get_inputs()[0].name: inp})
            seg = self.postprocess(outs[0], frame.shape)
            return seg
        except Exception as e:
            # if model fails, return None so caller can fallback
            print("FaceParser error:", e)
            return None
