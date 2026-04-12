# src/detector.py

import cv2


def detect_main_face(img_gray):
    """
    Detect the largest face in the image using Haar Cascade.
    Returns:
        (x, y, w, h) : bounding box of the main face
    """

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Haar cascade file not found.")

    # Scan the grayscale image for faces at multiple scales
    # scaleFactor=1.1  → shrink image by 10% at each scan pass
    # minNeighbors=5   → 5 overlapping detections must agree (reduces false hits)
    # minSize=(60,60)  → ignore any detection smaller than 60x60 pixels
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # No face found at all
    if len(faces) == 0:
        raise RuntimeError(
            "No face detected in the image.\n"
        )

    # Pick the largest face by area
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    print(f"[detector] Face found — x:{x} y:{y} w:{w} h:{h}  "
          f"| {len(faces)} face(s) detected total")

    return int(x), int(y), int(w), int(h)