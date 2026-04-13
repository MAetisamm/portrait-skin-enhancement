# src/mask.py

import cv2
import numpy as np


def build_skin_mask(img_ycrcb, face_box, landmarks,
                    lower_skin=None, upper_skin=None):
    """
    Build soft skin mask using dynamic skin color range.
    If lower/upper not provided, uses default medium skin range.
    """

    H, W = img_ycrcb.shape[:2]
    x, y, w, h = face_box

    # Use provided range or fall back to default
    if lower_skin is None:
        lower_skin = np.array([60, 120, 80],  dtype=np.uint8)
    if upper_skin is None:
        upper_skin = np.array([255, 180, 130], dtype=np.uint8)

    # 5a — Threshold skin color
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)

    # 5b — Restrict to face bounding box
    face_roi = np.zeros((H, W), dtype=np.uint8)
    face_roi[y:y+h, x:x+w] = 255
    skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_roi)

    # 5c — Carve out non-skin features
    def make_polygon(indices):
        return np.array(
            [landmarks[i] for i in indices],
            dtype=np.int32
        ).reshape(-1, 1, 2)

    cv2.fillPoly(skin_mask, [make_polygon(range(36, 42))], 0)
    cv2.fillPoly(skin_mask, [make_polygon(range(42, 48))], 0)
    cv2.fillPoly(skin_mask, [make_polygon(range(17, 22))], 0)
    cv2.fillPoly(skin_mask, [make_polygon(range(22, 27))], 0)
    cv2.fillPoly(skin_mask, [make_polygon(range(48, 68))], 0)

    # 5d — Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # 5e — Feather edges
    mask_float = skin_mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (31, 31), sigmaX=15)

    print(f"[mask] Coverage: {(skin_mask > 0).sum() / (H*W) * 100:.1f}%")
    return mask_float