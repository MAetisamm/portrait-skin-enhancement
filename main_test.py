import cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
original = cv2.imread('input/test9.jpg')
enhanced = cv2.imread('output/enhanced.jpg')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original', fontsize=14)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
plt.title('Enhanced', fontsize=14)
plt.axis('off')
plt.suptitle('Portrait Skin Enhancement — Before vs After',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/before_after.jpg', dpi=150, bbox_inches='tight')
print('Saved to assets/before_after.jpg')
