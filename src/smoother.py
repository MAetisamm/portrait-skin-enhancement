import cv2
def smooth_skin(img_bgr, face_box, radius=10, eps=0.04):
    """
    Smooth the face region using Guided Filter.

    Guided Filter smooths skin texture but preserves sharp
    edges like eyelids, lip boundaries and glasses frames.
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

    img_smooth = img_bgr.copy()
    img_smooth[y:y+h, x:x+w] = smoothed_face

    print(f"[smoother] Face region smoothed: {w}x{h} px")
    return img_smooth