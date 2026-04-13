import cv2
import numpy as np
def smooth_skin(img_bgr, face_box, radius=2, eps=0.02, strength=0.6):
    """
    Smooth the face region using Guided Filter.

    Guided Filter smooths skin texture but preserves sharp
    edges like eyelids, lip boundaries and glasses frames.
    strength - how much smoothing to apply
            0.0 = no smoothing (original)
            0.5 = 50% smooth 50% original
            1.0 = full smoothing

    """

    x, y, w, h = face_box
    face_crop = img_bgr[y:y+h, x:x+w]

    try:
        smoothed_face = cv2.ximgproc.guidedFilter(
            guide=face_crop,
            src=face_crop,
            radius=radius,
            eps=eps * 255 * 255
        )
        print(f"[smoother] Guided Filter applied")
        print(f"[smoother] radius={radius}  eps={eps}")

    except AttributeError:
        # Fallback if opencv-contrib is not installed
        print("[smoother] WARNING: Guided Filter not available")
        print("[smoother] Falling back to Bilateral Filter")
        smoothed_face = cv2.bilateralFilter(
            face_crop, d=9,
            sigmaColor=60,
            sigmaSpace=60
        )

    face_crop_f   = face_crop.astype(np.float32)
    smoothed_f    = smoothed_face.astype(np.float32)
    blended       = strength * smoothed_f + (1.0 - strength) * face_crop_f
    blended_face  = np.clip(blended, 0, 255).astype(np.uint8)

    img_smooth = img_bgr.copy()
    img_smooth[y:y+h, x:x+w] = blended_face

    print(f"[smoother] Face region smoothed: {w}x{h} px")
    return img_smooth