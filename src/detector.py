import cv2
import dlib

def detect_main_face(img_gray):
    """
    Detect face using Dlib HOG detector (handles angled faces).
    Falls back to Haar Cascade if Dlib finds nothing.
    """

    H, W = img_gray.shape[:2]

    img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    detector = dlib.get_frontal_face_detector()

    for upsample in [0, 1, 2]:
        dets = detector(img_rgb, upsample)
        if len(dets) > 0:
            print(f"[detector] Dlib found {len(dets)} face(s) "
                f"at upsample={upsample}")
            break

    if len(dets) > 0:
        # Pick largest detection
        det = max(dets, key=lambda d: d.width() * d.height())
        x, y = det.left(), det.top()
        w, h = det.width(), det.height()

    else:
        print("[detector] Dlib found nothing — trying Haar...")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = []
        for scale in [1.05, 1.1, 1.2, 1.3]:
            faces = face_cascade.detectMultiScale(
                img_gray,
                scaleFactor=scale,
                minNeighbors=3,
                minSize=(40, 40)
            )
            if len(faces) > 0:
                print(f"[detector] Haar found face at scale={scale}")
                break

        if len(faces) == 0:
            raise RuntimeError("No face detected in the image.")

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    print(f"[detector] Raw box  : x={x} y={y} w={w} h={h}")

    pad_x     = int(w * 0.15)   # 15% sides
    pad_y_top = int(h * 0.40)   # 40% top — covers forehead
    pad_y_bot = int(h * 0.10)   # 10% bottom — chin

    x = max(0, x - pad_x)
    y = max(0, y - pad_y_top)
    w = min(W - x, w + 2 * pad_x)
    h = min(H - y, h + pad_y_top + pad_y_bot)

    print(f"[detector] Final box: x={x} y={y} w={w} h={h}")
    print(f"[detector] Image size: {W}x{H}")

    return int(x), int(y), int(w), int(h)