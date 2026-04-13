import cv2
import numpy as np


def smooth_skin(img_bgr, face_box, mask_float,
                radius=3, eps=0.01, strength=0.5):
    """
    Smooth skin using Guided Filter.
    Mask applied so rectangle boundary is invisible.
    """

    x, y, w, h = face_box
    face_crop = img_bgr[y:y+h, x:x+w]
    face_mask = mask_float[y:y+h, x:x+w]

    try:
        smoothed = cv2.ximgproc.guidedFilter(
            guide=face_crop,
            src=face_crop,
            radius=radius,
            eps=eps * 255 * 255
        )
    except AttributeError:
        smoothed = cv2.bilateralFilter(
            face_crop, d=9,
            sigmaColor=60,
            sigmaSpace=60
        )

    # Blend smoothed with original using mask
    # Outside mask → keep original → no rectangle visible
    face_f    = face_crop.astype(np.float32)
    smooth_f  = smoothed.astype(np.float32)
    mask_3ch  = np.stack([face_mask]*3, axis=2)

    blended_f = (mask_3ch * strength * smooth_f +
                 (1 - mask_3ch * strength) * face_f)
    blended   = np.clip(blended_f, 0, 255).astype(np.uint8)

    img_smooth = img_bgr.copy()
    img_smooth[y:y+h, x:x+w] = blended

    print(f"[smoother] radius={radius} strength={strength}")
    return img_smooth