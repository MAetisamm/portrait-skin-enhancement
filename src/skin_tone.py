# src/skin_tone.py

import cv2
import numpy as np


def detect_skin_tone(img_bgr, face_box):
    """
    Detect skin tone and return tuned parameters.
    """

    x, y, w, h = face_box

    # Sample face center
    cx1 = x + int(w * 0.30)
    cx2 = x + int(w * 0.70)
    cy1 = y + int(h * 0.25)
    cy2 = y + int(h * 0.65)

    face_center  = img_bgr[cy1:cy2, cx1:cx2]
    center_ycrcb = cv2.cvtColor(face_center, cv2.COLOR_BGR2YCrCb)

    avg_Y  = float(np.mean(center_ycrcb[:, :, 0]))
    avg_Cr = float(np.mean(center_ycrcb[:, :, 1]))
    avg_Cb = float(np.mean(center_ycrcb[:, :, 2]))

    print(f"[skin_tone] Y={avg_Y:.1f}  Cr={avg_Cr:.1f}  Cb={avg_Cb:.1f}")

    # Classify tone
    if avg_Y > 160:
        tone = 'light'
    elif avg_Y > 100:
        tone = 'medium'
    else:
        tone = 'dark'

    print(f"[skin_tone] Detected: {tone.upper()}  (Y={avg_Y:.1f})")

    if tone == 'light':
        params = {
            'lower_skin' : np.array([80,  133,  85], dtype=np.uint8),
            'upper_skin' : np.array([255, 173, 125], dtype=np.uint8),
            'radius'     : 4,
            'eps'        : 0.01,
            'strength'   : 0.55,
        }

    elif tone == 'medium':
        params = {
            'lower_skin' : np.array([60,  120,  80], dtype=np.uint8),
            'upper_skin' : np.array([255, 180, 130], dtype=np.uint8),
            'radius'     : 3,
            'eps'        : 0.015,
            'strength'   : 0.5,
        }

    else:
        params = {
            'lower_skin' : np.array([30,  100,  70], dtype=np.uint8),
            'upper_skin' : np.array([255, 185, 140], dtype=np.uint8),
            'radius'     : 2,
            'eps'        : 0.02,
            'strength'   : 0.4,
        }

    print(f"[skin_tone]   radius={params['radius']}  "
          f"strength={params['strength']}  ")

    return tone, params