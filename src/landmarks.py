import dlib
import cv2
import numpy as np
import os

def get_facial_landmarks(img_bgr, face_box,
                        predictor_path="shape_predictor_68_face_landmarks.dat"):
    """
    Detect 68 landmark points on the face.

    Landmark groups:
        pts[0  - 16] -> jawline        (17 points)
        pts[17 - 21] -> left eyebrow   ( 5 points)
        pts[22 - 26] -> right eyebrow  ( 5 points)
        pts[27 - 35] -> nose           ( 9 points)
        pts[36 - 41] -> left eye       ( 6 points)
        pts[42 - 47] -> right eye      ( 6 points)
        pts[48 - 67] -> lips           (20 points)

    Returns:
        pts : list of 68 (x, y) tuples
    """

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(
            f"Landmark model not found: {predictor_path}"
        )

    predictor = dlib.shape_predictor(predictor_path)

    H, W = img_bgr.shape[:2]
    x, y, w, h = face_box

    shrink_x = int(w * 0.05)
    shrink_y = int(h * 0.05)

    rect = dlib.rectangle(
        left  = min(W-1, x + shrink_x),
        top   = min(H-1, y + shrink_y),
        right = min(W-1, x + w - shrink_x),
        bottom= min(H-1, y + h - shrink_y)
    )

    shape = predictor(img_bgr, rect)

    pts = []
    for i in range(68):
        point = shape.part(i)
        pts.append((point.x, point.y))

    out_of_bounds = sum(
        1 for px, py in pts
        if px < 0 or px >= W or py < 0 or py >= H
    )

    if out_of_bounds > 0:
        print(f"[landmarks] WARNING: {out_of_bounds} points "
            f"outside image bounds!")
    else:
        print("[landmarks] All 68 points within image bounds ✓")

    print("[landmarks] -----------------------------------")
    print("[landmarks] Jaw start      -> pts[0]  :", pts[0])
    print("[landmarks] Jaw end        -> pts[16] :", pts[16])
    print("[landmarks] Left brow      -> pts[17] :", pts[17],
        " pts[21]:", pts[21])
    print("[landmarks] Right brow     -> pts[22] :", pts[22],
        " pts[26]:", pts[26])
    print("[landmarks] Left eye       -> pts[36] :", pts[36],
        " pts[41]:", pts[41])
    print("[landmarks] Right eye      -> pts[42] :", pts[42],
        " pts[47]:", pts[47])
    print("[landmarks] Lip left       -> pts[48] :", pts[48])
    print("[landmarks] Lip right      -> pts[54] :", pts[54])
    print("[landmarks] -----------------------------------")
    print("[landmarks] Total points   :", len(pts))

    return pts