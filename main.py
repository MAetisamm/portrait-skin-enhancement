# main.py

import cv2
import matplotlib.pyplot as plt
from src.loader     import load_image
from src.detector   import detect_main_face
from src.landmarks  import get_facial_landmarks
from src.skin_tone  import detect_skin_tone
from src.mask       import build_skin_mask
from src.smoother   import smooth_skin
from src.tone       import enhance_tone
from src.blender    import blend_and_save

print("=" * 55)
print("  Portrait Skin Enhancement Pipeline")
print("=" * 55)

print("\n[STEP 1] Loading image...")
bgr, gray, ycrcb = load_image('input/test9.jpg')
img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print("  STATUS: OK")

print("\n[STEP 2] Detecting face...")
face_box = detect_main_face(gray)
x, y, w, h = face_box
print("  STATUS: OK")

print("\n[STEP 3] Detecting landmarks...")
pts = get_facial_landmarks(bgr, face_box)
print("  STATUS: OK")

print("\n[STEP 4] Detecting skin tone...")
tone, params = detect_skin_tone(bgr, face_box)
print("  STATUS: OK")

print("\n[STEP 5] Building skin mask...")
mask = build_skin_mask(ycrcb, face_box, pts,
                        lower_skin=params['lower_skin'],
                        upper_skin=params['upper_skin'])
print("  STATUS: OK")

# ── Step 6 — Smooth ───────────────────────────────────────
print("\n[STEP 6] Smoothing skin...")
img_smooth = smooth_skin(
    bgr, face_box,
    mask_float=mask,           # pass mask here
    radius    =params['radius'],
    eps       =params['eps'],
    strength  =params['strength']
)
print("  STATUS: OK")

print("\n[STEP 7] Tone enhancement...")
img_tone = enhance_tone(
    img_smooth, face_box, mask,
    brightness_boost=1.0
)
print("  STATUS: OK")

print("\n[STEP 8] Blending and saving...")
result = blend_and_save(bgr, img_tone, mask,
                        output_path='output/enhanced.jpg')
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
print("  STATUS: OK")

print("\n" + "=" * 55)
print(f"  Pipeline complete! Skin tone: {tone.upper()}")
print("=" * 55)

plt.figure()

plt.subplot(2, 4, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2, 4, 2)
img_face = img_rgb.copy()
cv2.rectangle(img_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(img_face)
plt.title("Face detected")
plt.axis('off')

plt.subplot(2, 4, 3)
img_lm = img_rgb.copy()
for px, py in pts:
    cv2.circle(img_lm, (px, py), 3, (255, 0, 0), -1)
plt.imshow(img_lm)
plt.title("68 landmarks")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(mask, cmap='gray')
plt.title("Skin mask")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(img_smooth, cv2.COLOR_BGR2RGB))
plt.title("Smoothed")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(img_tone, cv2.COLOR_BGR2RGB))
plt.title("Tone enhanced")
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(result_rgb)
plt.title(f"Final — {tone} skin")
plt.axis('off')

plt.suptitle(f"Portrait Skin Enhancement — {tone.upper()} skin tone",
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()