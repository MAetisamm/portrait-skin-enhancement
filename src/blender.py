# src/blender.py

import cv2
import numpy as np
import os


def blend_and_save(img_original, img_enhanced, mask_float, output_path):
    """
    Blend enhanced image with original using skin mask.
    Then save the final result to disk.
    """

    mask_3ch = np.stack([mask_float,
                        mask_float,
                        mask_float], axis=2)

    orig_f = img_original.astype(np.float32)
    enh_f  = img_enhanced.astype(np.float32)

    blended_f = mask_3ch * enh_f + (1.0 - mask_3ch) * orig_f
    result = np.clip(blended_f, 0, 255).astype(np.uint8)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ok = cv2.imwrite(output_path, result,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"Failed to save image to: {output_path}")
    print(f"[blender] Blending complete")
    print(f"[blender] Saved → {output_path}")
    return result