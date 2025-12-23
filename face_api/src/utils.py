# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np

def bgr2rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def cosine_similarity(vec, mat):
    """
    vec: (D,)
    mat: (N,D)
    return sims: (N,)  取值[-1,1]，越大越像
    """
    vec = vec.reshape(1, -1)
    # 确保都是单位向量（安全起见）
    v = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sims = (v @ m.T).reshape(-1)
    return sims

def list_images(folder):
    exts = ('.jpg','.jpeg','.png','.bmp','.webp')
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    return files

def load_db(emb_path, label_path):
    if not os.path.exists(emb_path) or not os.path.exists(label_path):
        raise FileNotFoundError("数据库不存在，请先运行 build_db.py 生成 embeddings.npy 和 labels.json")
    emb = np.load(emb_path)  # (N,512)
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)  # list[str]
    return emb, labels

def save_db(embeddings, labels, out_emb='outputs/embeddings.npy', out_labels='outputs/labels.json'):
    os.makedirs(os.path.dirname(out_emb), exist_ok=True)
    np.save(out_emb, embeddings)
    with open(out_labels, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
