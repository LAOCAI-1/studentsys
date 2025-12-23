# api_service.py (建议直接覆盖原文件再微调)
import os
import io
import json
import base64
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import yaml
import torch
import pymysql

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------ 解决 src/ 的导入 ------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"

import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detectors import FaceDetector
from align import align_face
from model import FaceEmbeddingModel
from utils import bgr2rgb, cosine_similarity  # 不再用 load_db/save_db 做主存储

app = FastAPI(title="Face API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Config ==========
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

if not CONFIG_PATH.exists():
    default_cfg = {"image_size": 160, "threshold": 0.55, "device": "cuda", "min_face_score": 0.70}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(default_cfg, f, allow_unicode=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

_cfg_device = str(CFG.get("device", "cpu")).lower()
if _cfg_device == "cuda" and torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

IMAGE_SIZE = int(CFG.get("image_size", 160))
THRESHOLD_DEFAULT = float(CFG.get("threshold", 0.55))
MIN_FACE_SCORE = float(CFG.get("min_face_score", 0.70))

DETECTOR = FaceDetector(device=DEVICE, min_face_score=MIN_FACE_SCORE)
EMB_MODEL = FaceEmbeddingModel(device=DEVICE)

# ========== MySQL Config ==========
DB_HOST = os.getenv("FACE_DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("FACE_DB_PORT", "3306"))
DB_USER = os.getenv("FACE_DB_USER", "root")
DB_PASS = os.getenv("FACE_DB_PASSWORD", "")
DB_NAME = os.getenv("FACE_DB_NAME", "studentsys")
DB_TABLE = os.getenv("FACE_DB_TABLE", "face_embeddings")

# 内存缓存（用于快速 verify）
DB_LOCK = threading.Lock()
CACHE_EMB = None        # shape (N, D), float32
CACHE_LABELS = []       # list[str]
CACHE_META = []         # list[dict] 可选，存 account_id/dim/model等

def _mysql_conn():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )

def _reload_from_mysql():
    """
    读取整库到内存：label + embedding(BLOB)。
    注意：embedding 必须是 float32 bytes（np.frombuffer 可还原）。
    """
    global CACHE_EMB, CACHE_LABELS, CACHE_META
    with _mysql_conn() as conn:
        with conn.cursor() as cur:
            # ⚠️ 这里的字段名需要你用 SHOW CREATE TABLE face_embeddings; 给我确认
            cur.execute(f"SELECT label, account_id, dim, model, embedding FROM {DB_TABLE}")
            rows = cur.fetchall()

    labels = []
    vecs = []
    meta = []
    for r in rows:
        blob = r.get("embedding", None)
        if blob is None:
            continue
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size == 0:
            continue
        # 保险：归一化（跟你原逻辑一致）
        v = v / (np.linalg.norm(v) + 1e-12)
        labels.append(r["label"])
        vecs.append(v.astype(np.float32))
        meta.append({
            "label": r["label"],
            "account_id": r.get("account_id"),
            "dim": int(r.get("dim") or v.size),
            "model": r.get("model"),
        })

    if len(vecs) == 0:
        CACHE_EMB = None
        CACHE_LABELS = []
        CACHE_META = []
        return 0

    # 如果维度不一致，直接报错更直观（避免 cosine_similarity 崩）
    d0 = vecs[0].size
    for v in vecs:
        if v.size != d0:
            raise RuntimeError(f"embedding dim mismatch: expect {d0}, got {v.size}")

    CACHE_EMB = np.stack(vecs, axis=0)  # (N, D)
    CACHE_LABELS = labels
    CACHE_META = meta
    return len(labels)

def _upsert_one_embedding(label: str, vec: np.ndarray, account_id: Optional[int], model: str = "face_api"):
    """
    不依赖 UNIQUE KEY：先查再 update/insert（最稳）。
    """
    vec = vec.astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    blob = vec.tobytes()
    dim = int(vec.size)

    with _mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {DB_TABLE} WHERE label=%s LIMIT 1", (label,))
            row = cur.fetchone()
            if row:
                cur.execute(
                    f"UPDATE {DB_TABLE} SET account_id=%s, dim=%s, model=%s, embedding=%s WHERE id=%s",
                    (account_id, dim, model, blob, row["id"])
                )
            else:
                cur.execute(
                    f"INSERT INTO {DB_TABLE} (account_id, label, dim, model, embedding) VALUES (%s,%s,%s,%s,%s)",
                    (account_id, label, dim, model, blob)
                )

def _bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片")
    return img

def _decode_base64_to_bgr(data_url: str) -> np.ndarray:
    if "," in data_url and data_url.strip().startswith("data:"):
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)
    return _bytes_to_bgr(img_bytes)

def _infer_one_image(bgr_img: np.ndarray, topk: int, thres: float):
    global CACHE_EMB, CACHE_LABELS
    if CACHE_EMB is None or len(CACHE_LABELS) == 0:
        raise RuntimeError("人脸库为空：请先 enroll，或检查 MySQL face_embeddings 是否有数据")

    img_rgb = bgr2rgb(bgr_img)
    faces = DETECTOR.detect_faces(img_rgb)
    if not faces:
        return {"ok": False, "faces": 0, "best_label": None, "best_score": 0.0, "topk": [], "message": "未检测到人脸"}

    best_overall = {"label": None, "score": -1.0}
    topk_list = []

    for f in faces:
        aligned = align_face(img_rgb, f["keypoints"], size=IMAGE_SIZE)
        vec = EMB_MODEL.get_embedding(aligned)  # (D,)
        sims = cosine_similarity(vec, CACHE_EMB)  # (N,)

        idx_sorted = np.argsort(-sims)[:max(1, topk)]
        cur_topk = [{"label": CACHE_LABELS[i], "score": float(sims[i])} for i in idx_sorted]
        topk_list.append(cur_topk)

        i_best = int(np.argmax(sims))
        s_best = float(sims[i_best])
        if s_best > best_overall["score"]:
            best_overall = {"label": CACHE_LABELS[i_best], "score": s_best}

    final_label = best_overall["label"] if best_overall["score"] >= thres else "Unknown"
    return {"ok": True, "faces": len(faces), "best_label": final_label, "best_score": best_overall["score"], "topk": topk_list}

# ========== Startup: load cache ==========
try:
    _reload_from_mysql()
except Exception as e:
    # 不要直接启动失败，先让 /health 可见错误
    print("[FaceAPI] MySQL preload failed:", e)

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "cuda_available": bool(torch.cuda.is_available()),
        "image_size": IMAGE_SIZE,
        "threshold_default": THRESHOLD_DEFAULT,
        "db_size": len(CACHE_LABELS),
        "mysql": {"host": DB_HOST, "port": DB_PORT, "db": DB_NAME, "table": DB_TABLE, "user": DB_USER},
    }

@app.post("/reload")
def reload_db():
    try:
        with DB_LOCK:
            n = _reload_from_mysql()
        return {"ok": True, "db_size": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== 兼容 Node：multipart/form-data =====
@app.post("/verify")
async def verify(
    file: UploadFile = File(None),
    image_base64: str = Form(None),
    topk: int = Form(1),
    threshold: float = Form(None),
):
    try:
        if file is None and not image_base64:
            raise ValueError("缺少 file 或 image_base64")
        raw = await file.read() if file is not None else None
        img_bgr = _bytes_to_bgr(raw) if raw is not None else _decode_base64_to_bgr(image_base64)
        thres = float(threshold) if threshold is not None else THRESHOLD_DEFAULT

        res = _infer_one_image(img_bgr, topk=int(topk or 1), thres=thres)

        # ✅ 兼容你网页显示用的字段名（避免 undefined / NaN）
        res["label"] = res.get("best_label")
        res["score"] = res.get("best_score")

        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/enroll")
async def enroll(
    label: str = Form(...),
    account_id: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None),  # 预留，不一定用
    files: List[UploadFile] = File(...),
):
    """
    Node /enroll-secure 会传：label + files[]
    这里做：
      - 多张图提取 embedding -> mean -> 写 MySQL face_embeddings
      - 同步刷新内存 cache，保证 verify 立刻可用
    """
    global CACHE_EMB, CACHE_LABELS
    try:
        if not label or not label.strip():
            raise ValueError("label 不能为空")
        if not files or len(files) == 0:
            raise ValueError("files 不能为空")

        vecs = []
        for uf in files:
            raw = await uf.read()
            img_bgr = _bytes_to_bgr(raw)
            img_rgb = bgr2rgb(img_bgr)
            faces = DETECTOR.detect_faces(img_rgb)
            if not faces:
                continue
            f = max(faces, key=lambda x: x["score"])
            aligned = align_face(img_rgb, f["keypoints"], size=IMAGE_SIZE)
            vec = EMB_MODEL.get_embedding(aligned)
            vecs.append(vec)

        if not vecs:
            raise ValueError("所有图片都未检测到可用人脸")

        mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)

        with DB_LOCK:
            _upsert_one_embedding(label.strip(), mean_vec, account_id, model="face_api")
            n = _reload_from_mysql()

        return {"ok": True, "label": label.strip(), "images_used": len(vecs), "db_size": n}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
