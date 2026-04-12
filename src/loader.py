import cv2
import os

def load_image(image_path):

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    h, w = img_bgr.shape[:2]
    print(f"[loader] Image loaded: {w}x{h} px  |  path: {image_path}")

    return img_bgr, img_gray, img_ycrcb