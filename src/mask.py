# src/mask.py

import cv2
import numpy as np


def build_skin_mask(img_ycrcb, face_box, landmarks,
                    lower_skin=None, upper_skin=None):

    H, W = img_ycrcb.shape[:2]
    x, y, w, h = face_box

    all_pts = list(landmarks)

    forehead_shift = int(h * 0.30)
    for i in range(17, 27):
        px, py = landmarks[i]
        all_pts.append((px, max(0, py - forehead_shift)))

    pts_np = np.array(all_pts, dtype=np.int32)
    hull   = cv2.convexHull(pts_np)

    skin_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(skin_mask, [hull], 255)

    def poly(indices):
        return np.array(
            [landmarks[i] for i in indices],
            dtype=np.int32
        ).reshape(-1, 1, 2)

    # Left eye
    cv2.fillPoly(skin_mask, [poly(range(36, 42))], 0)
    # Right eye
    cv2.fillPoly(skin_mask, [poly(range(42, 48))], 0)
    # Lips
    cv2.fillPoly(skin_mask, [poly(range(48, 68))], 0)

    mask_float = skin_mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (31, 31), sigmaX=10)

    coverage = (skin_mask > 0).sum() / (H * W) * 100
    print(f"[mask] Coverage: {coverage:.1f}%")

    return mask_float