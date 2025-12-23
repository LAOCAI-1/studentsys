# -*- coding: utf-8 -*-
"""
两种用法：
1) 单张图片（保存结果到 outputs/result.png）：
   python src/inference.py --image path/to/img.jpg

2) 摄像头（保存结果到 outputs/cam_out.mp4）：
   python src/inference.py --camera 0 --headless
"""
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")

import argparse
import yaml
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path  # ← 新增：用于相对路径

from detectors import FaceDetector
from align import align_face
from model import FaceEmbeddingModel
from utils import bgr2rgb, load_db, cosine_similarity

def draw_one(frame_bgr, box, label, score):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} {score:.2f}"
    cv2.putText(frame_bgr, text, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def infer_on_frame(frame_bgr, det, emb_model, db_emb, db_labels, image_size, threshold):
    img_rgb = bgr2rgb(frame_bgr)
    faces = det.detect_faces(img_rgb)
    results = []
    for f in faces:
        aligned = align_face(img_rgb, f['keypoints'], size=image_size)
        vec = emb_model.get_embedding(aligned)
        sims = cosine_similarity(vec, db_emb)
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        label = db_labels[idx] if best >= threshold else "Unknown"
        results.append((f['box'], label, best))
    return results

def safe_save_image(img_bgr, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img_bgr)
    print(f"[HEADLESS] 无法使用 GUI，已将结果保存到：{os.path.abspath(out_path)}")

def run_image(path, det, emb_model, db_emb, db_labels, image_size, threshold, out_dir="outputs"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    results = infer_on_frame(img, det, emb_model, db_emb, db_labels, image_size, threshold)
    for (box, label, score) in results:
        draw_one(img, box, label, score)
    # 仅保存文件（不弹窗）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"result_{ts}.png")
    safe_save_image(img, out_path)

def run_camera(index, det, emb_model, db_emb, db_labels, image_size, threshold, headless=False, out_dir="outputs"):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头：{index}")

    writer = None
    if headless:
        os.makedirs(out_dir, exist_ok=True)
        # 预读一帧确定尺寸
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("摄像头读帧失败")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_video = os.path.join(out_dir, f"cam_out_{ts}.mp4")
        writer = cv2.VideoWriter(out_video, fourcc, 20.0, (w, h))
        # 先处理预读帧
        results = infer_on_frame(frame, det, emb_model, db_emb, db_labels, image_size, threshold)
        for (box, label, score) in results:
            draw_one(frame, box, label, score)
        writer.write(frame)
        print(f"[HEADLESS] 正在录制到：{os.path.abspath(out_video)}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = infer_on_frame(frame, det, emb_model, db_emb, db_labels, image_size, threshold)
        for (box, label, score) in results:
            draw_one(frame, box, label, score)

        if headless:
            writer.write(frame)
        else:
            # 有 GUI 环境才显示
            cv2.imshow("Face Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print("[HEADLESS] 已完成录制（按 Ctrl+C 结束进程或关闭终端）")
    else:
        cv2.destroyAllWindows()

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = cfg.get('device', 'cpu')
    image_size = int(cfg.get('image_size', 112))
    threshold = float(cfg.get('threshold', 0.55))
    min_face_score = float(cfg.get('min_face_score', 0.9))

    db_emb, db_labels = load_db(args.embeddings, args.labels)
    det = FaceDetector(device=device, min_face_score=min_face_score)
    emb_model = FaceEmbeddingModel(device=device)

    if args.image:
        run_image(args.image, det, emb_model, db_emb, db_labels, image_size, threshold)
    else:
        run_camera(args.camera, det, emb_model, db_emb, db_labels, image_size, threshold, headless=args.headless)

if __name__ == "__main__":
    # 你已经在用绝对路径，这里保持一致
    # （已替换为相对路径，自动定位项目根目录）
    root = Path(__file__).resolve().parents[1]
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=str(root / "configs" / "config.yaml"))
    parser.add_argument("--embeddings", default=str(root / "outputs" / "embeddings.npy"))
    parser.add_argument("--labels",     default=str(root / "outputs" / "labels.json"))
    parser.add_argument("--image",      default=None, help="单张图片路径")
    parser.add_argument("--camera",     type=int, default=0, help="摄像头索引")
    parser.add_argument("--headless",   action="store_true", help="无GUI环境：不弹窗，保存图片/视频到 outputs/")
    args = parser.parse_args()
    main(args)
