# main.py

import cv2
import matplotlib.pyplot as plt
from src.loader import load_image
from src.detector import detect_main_face
from src.landmarks import get_facial_landmarks
from src.mask import build_skin_mask
from src.smoother import smooth_skin

bgr, gray, ycrcb = load_image('input/test1.jpg')
img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

face_box = detect_main_face(gray)
x, y, w, h = face_box

pts = get_facial_landmarks(bgr, face_box)

mask = build_skin_mask(ycrcb, face_box, pts)

img_smooth = smooth_skin(bgr, face_box)
img_smooth_rgb = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB)



img_face = img_rgb.copy()
cv2.rectangle(img_face, (x, y), (x+w, y+h), (0, 255, 0), 5)



# ── Draw landmarks on a copy ──────────────────────────────
img_landmarks = img_rgb.copy()
for px, py in pts:
    cv2.circle(img_landmarks, (px, py), 2, (255, 0, 0), 8)

plt.subplot(3, 1, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(img_face)
plt.title("Face detected")
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(img_landmarks)
plt.title("68 landmarks")
plt.axis('off')

plt.subplot(3, 2, 1)
plt.imshow(mask, cmap='gray')
plt.title("Skin mask")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(img_smooth_rgb)
plt.title("Smoothed skin")
plt.axis('off')


plt.tight_layout()
plt.show()