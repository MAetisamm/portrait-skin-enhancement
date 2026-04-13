# src/mask.py

import cv2
import numpy as np


def build_skin_mask(img_ycrcb, face_box, landmarks,
                    lower_skin=None, upper_skin=None):
    """
    Build soft skin mask using extended convex hull
    that includes the forehead area above eyebrows.
    """

    H, W = img_ycrcb.shape[:2]
    x, y, w, h = face_box

    if lower_skin is None:
        lower_skin = np.array([60, 120, 80],  dtype=np.uint8)
    if upper_skin is None:
        upper_skin = np.array([255, 180, 130], dtype=np.uint8)

    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    all_pts = list(landmarks)
    forehead_shift = int(h * 0.40)  # shift up by 40% of face height

    for i in range(17, 27):  # both eyebrows
        px, py = landmarks[i]
        # Push point upward
        new_py = max(0, py - forehead_shift)
        all_pts.append((px, new_py))

    nose_top_x, nose_top_y = landmarks[27]
    all_pts.append((nose_top_x, max(0, nose_top_y - forehead_shift)))

    all_pts_np = np.array(all_pts, dtype=np.int32)
    hull = cv2.convexHull(all_pts_np)

    face_shape_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(face_shape_mask, [hull], 255)

    skin_mask = cv2.bitwise_and(skin_mask, skin_mask,
                                mask=face_shape_mask)

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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    mask_float = skin_mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (51, 51), sigmaX=20)

    coverage = (skin_mask > 0).sum() / (H * W) * 100
    print(f"[mask] Coverage: {coverage:.1f}%")

    return mask_float