# src/tone.py

import cv2
import numpy as np


def enhance_tone(img_bgr, face_box, mask_float, brightness_boost=1.0):
    """
    img_bgr          - smoothed image from smoother.py
    face_box         - (x, y, w, h) face location
    mask_float       - skin mask 0.0 to 1.0 from mask.py
    brightness_boost - how much to brighten
                    1.0 = normal
                    1.5 = 50% more bright
                    2.0 = double brightness
    """

    x, y, w, h = face_box
    face_crop = img_bgr[y:y+h, x:x+w]
    face_mask = mask_float[y:y+h, x:x+w]

    face_mask = cv2.GaussianBlur(face_mask, (51, 51), sigmaX=20)

    # Convert to YCrCb
    face_ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
    Y = face_ycrcb[:, :, 0].astype(np.float32)

    # Get skin pixels only
    skin_pixels = Y[face_mask > 0.5]

    if len(skin_pixels) == 0:
        print("[tone] No skin pixels — skipping")
        return img_bgr.copy()

    # Find brightness range
    p_low  = np.percentile(skin_pixels, 5)
    p_high = np.percentile(skin_pixels, 95)
    p_mean = np.mean(skin_pixels)

    print(f"[tone] Skin Y — low:{p_low:.1f}  "
        f"mean:{p_mean:.1f}  high:{p_high:.1f}")

    target_high = min(240.0 * brightness_boost, 255.0)
    target_low  = max(p_low * 0.85, 15)

    if p_high == p_low:
        print("[tone] Flat image — skipping")
        return img_bgr.copy()

    # Stretch formula
    Y_stretched = (Y - p_low) / (p_high - p_low) * (target_high - target_low) + target_low
    Y_stretched = np.clip(Y_stretched, 0, 255)

    # Apply only to skin pixels
    Y_blended = face_mask * Y_stretched + (1.0 - face_mask) * Y
    face_ycrcb[:, :, 0] = np.clip(Y_blended, 0, 255).astype(np.uint8)

    enhanced_face = cv2.cvtColor(face_ycrcb, cv2.COLOR_YCrCb2BGR)

    img_tone = img_bgr.copy()
    img_tone[y:y+h, x:x+w] = enhanced_face

    print(f"[tone] target_high={target_high:.1f}  "
        f"boost={brightness_boost}")

    return img_tone