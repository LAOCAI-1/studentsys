# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 参考关键点（112x112），选用三点相似变换（双眼+鼻尖）
# 你也可以改成五点/仿射-相似 以获得更稳的对齐
REFERENCE_LANDMARKS_3POINT_112 = np.float32([
    [38.2946, 51.6963],  # left_eye
    [73.5318, 51.5014],  # right_eye
    [56.0252, 71.7366],  # nose
])

def align_face(img_rgb, keypoints, size=112):
    """
    使用 3 个点（左右眼+鼻尖）做相似变换，将人脸对齐到 size x size
    keypoints: dict with 'left_eye','right_eye','nose'
    """
    dst = REFERENCE_LANDMARKS_3POINT_112.copy()
    if size != 112:
        scale = float(size) / 112.0
        dst *= scale

    src = np.float32([
        [keypoints['left_eye'][0],  keypoints['left_eye'][1]],
        [keypoints['right_eye'][0], keypoints['right_eye'][1]],
        [keypoints['nose'][0],      keypoints['nose'][1]],
    ])

    # 估计相似变换矩阵
    M = cv2.estimateAffinePartial2D(src[None, ...], dst[None, ...], method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img_rgb, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return aligned
