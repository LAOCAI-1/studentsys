# -*- coding: utf-8 -*-
# face_api/src/model.py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from facenet_pytorch import InceptionResnetV1


class FaceEmbeddingModel:
    """
    使用 facenet-pytorch 的 InceptionResnetV1 (VGGFace2 预训练) 输出 512 维 embedding
    支持：
      - 指定 device：'cpu' / 'cuda' / 'cuda:0'
      - 指定输入尺寸 image_size（默认 112；模型也兼容 160 等）
      - 单张/批量提取（新增）：embed_one / embed_one_torch / embed_batch_np
      - 保留旧接口：
            get_embedding(img_rgb_112) -> np.ndarray(512,)
            embed_batch(faces_112_list) -> torch.Tensor[N,512] (在 CPU 上)
    """

    def __init__(self, device: str = "cpu", image_size: int = 112):
        # —— 设备选择（保持安全兜底）——
        want_cuda = isinstance(device, str) and device.startswith("cuda")
        if want_cuda and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.input_size = int(image_size)

        # pretrained='vggface2'：适合通用人脸特征
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    # ---------- 内部工具 ----------
    def _ensure_size_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        """确保输入为 BGR，且大小为 self.input_size x self.input_size"""
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return None
        h, w = img_bgr.shape[:2]
        if h != self.input_size or w != self.input_size:
            img_bgr = cv2.resize(
                img_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
            )
        return img_bgr

    def _bgr_to_tensor01(self, img_bgr_sz: np.ndarray) -> torch.Tensor:
        """
        BGR(uint8, HxWx3, 0~255) -> RGB(float32, 0~1), CHW tensor
        """
        img_rgb = img_bgr_sz[..., ::-1].astype(np.float32) / 255.0
        t = torch.from_numpy(img_rgb).permute(2, 0, 1)  # 3xHxW
        return t

    # ---------- 新增：对外便捷 API ----------
    def embed_one(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        输入：BGR 图 (HxWx3, uint8)，尺寸可不为 input_size，将自动缩放；
        输出：L2 归一化的 512 维 (np.float32,)
        """
        img_bgr = self._ensure_size_bgr(img_bgr)
        if img_bgr is None:
            return np.zeros((512,), dtype=np.float32)

        x = self._bgr_to_tensor01(img_bgr).unsqueeze(0).to(self.device)  # 1x3xHxW
        with torch.no_grad():
            emb = self.net(x)                 # (1,512)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def embed_one_torch(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        同 embed_one，但返回 torch.Tensor[512]（在 CPU 上）
        """
        img_bgr = self._ensure_size_bgr(img_bgr)
        if img_bgr is None:
            return torch.zeros((512,), dtype=torch.float32)

        x = self._bgr_to_tensor01(img_bgr).unsqueeze(0).to(self.device)  # 1x3xHxW
        with torch.no_grad():
            emb = self.net(x)               # (1,512)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).detach().cpu()

    def embed_batch_np(self, faces_list: list) -> np.ndarray:
        """
        新增：批量（任意尺寸 BGR/RGB 皆可传，内部会 resize）-> np.ndarray[N,512]
        """
        tensors = []
        for img in faces_list:
            if img is None or getattr(img, "size", 0) == 0:
                continue
            # 若是 RGB，也能用：BGR/RGB 统一按 BGR 视角 resize + 反转到 RGB
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                img_bgr = img
                # 尝试判断是否是 RGB：这里简单处理——无论 BGR/RGB，反转一次对 RGB/BGR 成对称操作
                img_bgr = self._ensure_size_bgr(img_bgr)
                if img_bgr is None:
                    continue
                tensors.append(self._bgr_to_tensor01(img_bgr))
        if not tensors:
            return np.zeros((0, 512), dtype=np.float32)

        x = torch.stack(tensors, dim=0).to(self.device)  # Nx3xHxW
        with torch.no_grad():
            emb = self.net(x)               # (N,512)
            emb = F.normalize(emb, p=2, dim=1)

        return emb.detach().cpu().numpy().astype(np.float32)

    # ---------- 保留旧接口 ----------
    def get_embedding(self, img_rgb_112: np.ndarray) -> np.ndarray:
        """
        旧版接口兼容：
          - 期望传入 RGB(112x112)，这里内部转换成 BGR 再走 embed_one
        返回：np.ndarray(512,)
        """
        if img_rgb_112 is None or getattr(img_rgb_112, "size", 0) == 0:
            return np.zeros((512,), dtype=np.float32)
        # RGB -> BGR
        img_bgr = img_rgb_112[..., ::-1]
        return self.embed_one(img_bgr)

    def embed_batch(self, faces_112_list):
        """
        【旧接口】与历史代码完全兼容：
          输入: faces_112_list = [np.ndarray(H,W,3), ...]  每张通常已对齐并缩放到 112x112
          注: 允许传入 BGR 或 RGB，这里统一转为 RGB，再归一化到 [0,1]，批量前向，L2 normalize。
          返回: torch.Tensor [N, 512] (在 CPU 上)
        """
        if not faces_112_list:
            return torch.empty((0, 512), dtype=torch.float32)

        arrs = []
        for face in faces_112_list:
            if face is None:
                continue
            if isinstance(face, np.ndarray) and face.ndim == 3 and face.shape[2] == 3:
                # 这里按“BGR->RGB”处理；若已是 RGB，反转一次对称也可用
                rgb = face[..., ::-1]
                # 若尺寸不是 input_size，也做一次 resize，保证兼容
                if rgb.shape[0] != self.input_size or rgb.shape[1] != self.input_size:
                    rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
                x = (rgb.astype(np.float32) / 255.0)  # [0,1]
                arrs.append(np.transpose(x, (2, 0, 1)))  # 3xHxW

        if not arrs:
            return torch.empty((0, 512), dtype=torch.float32)

        x = torch.from_numpy(np.stack(arrs, axis=0))  # Nx3xHxW
        x = x.to(self.device)
        with torch.no_grad():
            emb = self.net(x)                    # Nx512
            emb = F.normalize(emb, p=2, dim=1)   # L2 归一
        return emb.cpu()
