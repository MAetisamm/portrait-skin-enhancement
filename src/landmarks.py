
import dlib
import cv2
import numpy as np
import os


def get_facial_landmarks(img_bgr, face_box, predictor_path="shape_predictor_68_face_landmarks.dat"):
    """
    Detect 68 landmark points on the face.

    These points mark the exact boundaries of:
        pts[0  - 16] → jawline
        pts[17 - 21] → left eyebrow
        pts[22 - 26] → right eyebrow
        pts[27 - 35] → nose
        pts[36 - 41] → left eye
        pts[42 - 47] → right eye
        pts[48 - 67] → lips

    Returns:
        pts : list of 68 (x, y) tuples
    """

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(
            f"Landmark model not found: {predictor_path}\n"
            "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )

    predictor = dlib.shape_predictor(predictor_path)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    x, y, w, h = face_box

    rect = dlib.rectangle(
        left=x,
        top=y,
        right=x + w,
        bottom=y + h
    )

    # Run the predictor returns object with 68 points
    shape = predictor(img_rgb, rect)

    pts = []
    for i in range(68):
        point = shape.part(i)
        pts.append((point.x, point.y))

    print(f"[landmarks] 68 points detected successfully")
    print(f"[landmarks] -----------------------------------")

    # Jawline — pts 0 to 16
    print(f"[landmarks] Jaw start      → pts[0]  : {pts[0]}")
    print(f"[landmarks] Jaw middle     → pts[8]  : {pts[8]}")
    print(f"[landmarks] Jaw end        → pts[16] : {pts[16]}")

    # Eyebrows — pts 17 to 26
    print(f"[landmarks] Left brow      → pts[17] : {pts[17]}  pts[21]: {pts[21]}")
    print(f"[landmarks] Right brow     → pts[22] : {pts[22]}  pts[26]: {pts[26]}")

    # Nose — pts 27 to 35
    print(f"[landmarks] Nose bridge    → pts[27] : {pts[27]}")
    print(f"[landmarks] Nose tip       → pts[30] : {pts[30]}")
    print(f"[landmarks] Nose base      → pts[35] : {pts[35]}")

    # Eyes — pts 36 to 47
    print(f"[landmarks] Left eye       → pts[36] : {pts[36]}  pts[39]: {pts[39]}")
    print(f"[landmarks] Right eye      → pts[42] : {pts[42]}  pts[45]: {pts[45]}")

    # Lips — pts 48 to 67
    print(f"[landmarks] Lip left       → pts[48] : {pts[48]}")
    print(f"[landmarks] Lip right      → pts[54] : {pts[54]}")
    print(f"[landmarks] Lip top        → pts[51] : {pts[51]}")
    print(f"[landmarks] Lip bottom     → pts[57] : {pts[57]}")
    print(f"[landmarks] -----------------------------------")
    print(f"[landmarks] Total points   : {len(pts)}")

    return pts