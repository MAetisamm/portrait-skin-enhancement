import dlib
import cv2
import os


def get_facial_landmarks(img_bgr, face_box, predictor_path="shape_predictor_68_face_landmarks.dat"):
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
        raise FileNotFoundError(f"Landmark model not found: {predictor_path}")

    predictor = dlib.shape_predictor(predictor_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x, y, w, h = face_box
    rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
    shape = predictor(img_rgb, rect)

    pts = []
    for i in range(68):
        point = shape.part(i)
        pts.append((point.x, point.y))

    # Print all landmark groups to verify each region
    print("[landmarks] 68 points detected successfully")
    print("[landmarks] -----------------------------------")
    print("[landmarks] Jaw start      -> pts[0]  :", pts[0])
    print("[landmarks] Jaw middle     -> pts[8]  :", pts[8])
    print("[landmarks] Jaw end        -> pts[16] :", pts[16])
    print("[landmarks] Left brow      -> pts[17] :", pts[17], " pts[21]:", pts[21])
    print("[landmarks] Right brow     -> pts[22] :", pts[22], " pts[26]:", pts[26])
    print("[landmarks] Nose bridge    -> pts[27] :", pts[27])
    print("[landmarks] Nose tip       -> pts[30] :", pts[30])
    print("[landmarks] Nose base      -> pts[35] :", pts[35])
    print("[landmarks] Left eye       -> pts[36] :", pts[36], " pts[41]:", pts[41])
    print("[landmarks] Right eye      -> pts[42] :", pts[42], " pts[47]:", pts[47])
    print("[landmarks] Lip left       -> pts[48] :", pts[48])
    print("[landmarks] Lip right      -> pts[54] :", pts[54])
    print("[landmarks] Lip top        -> pts[51] :", pts[51])
    print("[landmarks] Lip bottom     -> pts[57] :", pts[57])
    print("[landmarks] Lip last       -> pts[67] :", pts[67])
    print("[landmarks] -----------------------------------")
    print("[landmarks] Total points   :", len(pts))

    return pts
