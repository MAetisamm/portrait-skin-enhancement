# src/mask.py

import cv2
import numpy as np


def build_skin_mask(img_ycrcb, face_box, landmarks):
    """
    Build a soft skin mask — 1.0 where skin is, 0.0 everywhere else.
    """

    H, W = img_ycrcb.shape[:2]
    x, y, w, h = face_box


    lower_skin = np.array([0,   133, 77],  dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)

    # This prevents detecting skin on neck, shoulders, others
    face_roi = np.zeros((H, W), dtype=np.uint8)
    face_roi[y:y+h, x:x+w] = 255
    skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_roi)

    def make_polygon(indices):
        return np.array(
            [landmarks[i] for i in indices],
            dtype=np.int32
        ).reshape(-1, 1, 2)

    # Left eye
    cv2.fillPoly(skin_mask, [make_polygon(range(36, 42))], 0)

    # Right eye
    cv2.fillPoly(skin_mask, [make_polygon(range(42, 48))], 0)

    # Left eyebrow
    cv2.fillPoly(skin_mask, [make_polygon(range(17, 22))], 0)

    # Right eyebrow
    cv2.fillPoly(skin_mask, [make_polygon(range(22, 27))], 0)

    # Lips
    cv2.fillPoly(skin_mask, [make_polygon(range(48, 68))], 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    mask_float = skin_mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (31, 31), sigmaX=15)

    print("[mask] Skin mask built successfully")
    print(f"[mask] Skin pixels  : {(skin_mask > 0).sum():,}")
    print(f"[mask] Total pixels : {H * W:,}")
    print(f"[mask] Coverage     : {(skin_mask > 0).sum() / (H * W) * 100:.1f}%")

    return mask_float