# -*- coding: utf-8 -*-
"""
从 data/known/ 读取每个人的多张照片：
1) 人脸检测 + 对齐
2) 计算每张 embedding
3) 对同一个人求均值向量（再L2归一化）
4) 保存 outputs/embeddings.npy 与 outputs/labels.json
"""
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")

import os
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path

# ✅ 与 main.py 保持一致的包结构
from src.detectors import FaceDetector
from src.align import align_face
from src.model import FaceEmbeddingModel
from src.utils import bgr2rgb, list_images, save_db  # 假定 utils 在 src 下

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    image_size = int(cfg.get('image_size', 112))
    device = cfg.get('device', 'cpu')
    min_face_score = float(cfg.get('min_face_score', 0.9))

    det = FaceDetector(device=device, min_face_score=min_face_score)
    emb_model = FaceEmbeddingModel(device=device, image_size=image_size)

    person_dirs = [d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))]
    person_dirs.sort()

    all_embeddings = []
    all_labels = []

    for person in person_dirs:
        folder = os.path.join(args.data_dir, person)
        imgs = list_images(folder)
        if not imgs:
            print(f"[WARN] {person} 没有图片，跳过")
            continue

        person_vecs = []
        for p in imgs:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                print(f"[WARN] 读取失败：{p}")
                continue
            img_rgb = bgr2rgb(img_bgr)

            # ✅ 一次就取最高分人脸（含关键点）
            faces = det.detect_faces(img_rgb)
            if not faces:
                print(f"[WARN] 未检测到人脸：{p}")
                continue
            f = max(faces, key=lambda x: x['score'])

            # ✅ 对齐失败兜底 bbox
            aligned = align_face(
                img_rgb, f['keypoints'], size=image_size,
                fallback_bbox=f['box']
            )
            vec = emb_model.get_embedding(aligned)  # 兼容老接口，内部转 BGR -> 新实现
            person_vecs.append(vec)

        if not person_vecs:
            print(f"[WARN] {person} 没有可用人脸，跳过")
            continue

        mean_vec = np.mean(np.stack(person_vecs, axis=0), axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
        all_embeddings.append(mean_vec.astype(np.float32))
        all_labels.append(person)
        print(f"[OK] {person}: {len(person_vecs)} 张，已入库")

    if not all_embeddings:
        raise RuntimeError("没有任何有效的人脸数据，请检查 data/known/*")

    embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
    save_db(embeddings, all_labels, out_emb=args.out_embeddings, out_labels=args.out_labels)
    print(f"\n完成：保存 {len(all_labels)} 人的模板到：\n{args.out_embeddings}\n{args.out_labels}")

if __name__ == "__main__":
    # ✅ root 设为 face_api 目录，保证和 main.py 的输出一致
    root = Path(__file__).resolve().parent
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "known").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       default=str(root / "data" / "known"), help="人脸库根目录（按人名分文件夹）")
    parser.add_argument("--config",         default=str(root / "configs" / "config.yaml"))
    parser.add_argument("--out-embeddings", default=str(root / "outputs" / "embeddings.npy"))
    parser.add_argument("--out-labels",     default=str(root / "outputs" / "labels.json"))
    args = parser.parse_args()
    main(args)
