# -*- coding: utf-8 -*-
import numpy as np
from facenet_pytorch import MTCNN
import cv2
import numpy as np
class FaceDetector:
    """
    封装 MTCNN：给定 BGR 或 RGB 图像，返回人脸框与关键点
    输出格式： list[ { 'box': (x1,y1,x2,y2), 'keypoints': {'left_eye':(x,y), 'right_eye':(x,y), 'nose':(x,y), 'mouth_left':(x,y), 'mouth_right':(x,y)}, 'score': float } ]
    """
    def __init__(self, device='cpu', min_face_score=0.9):
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.min_face_score = float(min_face_score)

    def detect_faces(self, img_rgb):
        # img_rgb: HxWx3 (RGB)
        boxes, probs, landmarks = self.mtcnn.detect(img_rgb, landmarks=True)
        results = []
        if boxes is None:
            return results
        for i in range(len(boxes)):
            score = float(probs[i]) if probs is not None else 1.0
            if score < self.min_face_score:
                continue
            box = boxes[i].astype(int)  # x1,y1,x2,y2
            lm = landmarks[i]
            kp = {
                'left_eye':  (float(lm[0][0]), float(lm[0][1])),
                'right_eye': (float(lm[1][0]), float(lm[1][1])),
                'nose':      (float(lm[2][0]), float(lm[2][1])),
                'mouth_left':(float(lm[3][0]), float(lm[3][1])),
                'mouth_right':(float(lm[4][0]), float(lm[4][1])),
            }
            results.append({'box': tuple(box.tolist()), 'keypoints': kp, 'score': score})
        return results
    
    def detect(self, img_bgr):
        """
        兼容 main.py：输入 OpenCV BGR，返回 (boxes, probs)
        boxes: Nx4 float32, probs: N float32；若无结果返回 (None, None)
        """
        if img_bgr is None or not hasattr(img_bgr, "shape") or img_bgr.size == 0:
            return None, None

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # landmarks=False 就只要框和分数
        boxes, probs = self.mtcnn.detect(img_rgb, landmarks=False)
        if boxes is None or len(boxes) == 0:
            return None, None

        boxes = np.asarray(boxes, dtype=np.float32)
        probs = np.asarray(probs, dtype=np.float32)

        # 依据 min_face_score 进行一次轻过滤（与 main.py 的二次校验兼容）
        if self.min_face_score is not None:
            keep = probs >= float(self.min_face_score)
            if not np.any(keep):
                return None, None
            boxes = boxes[keep]
            probs = probs[keep]

        return boxes, probs

    def detect_one(self, img_bgr):
        """
        取最高分人脸，返回 (box, prob, idx)；无结果返回 (None, None, None)
        """
        boxes, probs = self.detect(img_bgr)
        if boxes is None:
            return None, None, None
        i = int(np.argmax(probs))
        return boxes[i], float(probs[i]), i