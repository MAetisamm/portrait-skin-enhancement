"""
Portrait Skin Enhancement — Gradio Web UI
Run: python app.py
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import sys

# ── Make sure the src/ modules are importable ─────────────────────────────
# Put this file in the project root (same level as src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from loader     import load_image
from detector   import detect_main_face
from landmarks  import get_facial_landmarks
from skin_tone  import detect_skin_tone
from mask       import build_skin_mask
from smoother   import smooth_skin
from tone       import enhance_tone
from blender    import blend_and_save

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    input_image,           # numpy array RGB from Gradio
    radius: int,
    strength: float,
    brightness_boost: float,
    auto_params: bool,
):
    """Core processing function called by Gradio."""

    if input_image is None:
        raise gr.Error("Please upload a portrait photo first.")

    # Gradio passes images as RGB numpy arrays.
    # Convert to BGR for all OpenCV operations.
    img_bgr_input = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Save as lossless PNG -- NOT jpg.
    # Bug that was causing blur: saving as .jpg here applied a second lossy
    # compression pass before processing even started, softening the image.
    # Gradio also internally converts uploads to webp; PNG sidesteps that.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, img_bgr_input)   # lossless, no params needed

    try:
        # Step 1: Load (gives us fresh bgr, gray, ycrcb from lossless source)
        img_bgr, img_gray, img_ycrcb = load_image(tmp_path)

        # Step 2: Detect face
        face_box = detect_main_face(img_gray)

        # Step 3: 68-point landmarks
        landmarks = get_facial_landmarks(img_bgr, face_box,
                                         predictor_path=PREDICTOR_PATH)

        # Step 4: Skin tone classification
        tone, auto = detect_skin_tone(img_bgr, face_box)

        # Step 5: Parameter selection
        if auto_params:
            r   = auto["radius"]
            s   = auto["strength"]
            eps = auto["eps"]
        else:
            r   = radius
            s   = strength
            eps = 0.01

        # Step 6: Skin mask
        mask_float = build_skin_mask(img_ycrcb, face_box, landmarks,
                                     lower_skin=auto["lower_skin"],
                                     upper_skin=auto["upper_skin"])

        # Step 7: Edge-preserving smoothing (Guided Filter)
        img_smooth = smooth_skin(img_bgr, face_box, mask_float,
                                 radius=r, eps=eps, strength=s)

        # Step 8: Luminance / tone enhancement
        img_tone = enhance_tone(img_smooth, face_box, mask_float,
                                brightness_boost=brightness_boost)

        # Step 9: Alpha composite + save.
        # Bug that was causing blur: the original blender.blend_and_save forces
        # JPEG quality 95 which introduces ringing/softness artifacts.
        # We replicate the same Porter-Duff composite math and save as PNG.
        mask_3ch   = np.stack([mask_float] * 3, axis=2)
        orig_f     = img_bgr.astype(np.float32)
        enh_f      = img_tone.astype(np.float32)
        blended_f  = mask_3ch * enh_f + (1.0 - mask_3ch) * orig_f
        result_bgr = np.clip(blended_f, 0, 255).astype(np.uint8)

        out_path = tmp_path.replace(".png", "_enhanced.png")
        cv2.imwrite(out_path, result_bgr)   # lossless PNG -- sharp download

        # BGR -> RGB for Gradio
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        tone_label = f"Detected skin tone: {tone.upper()}"
        return result_rgb, out_path, tone_label

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0e0e10 !important;
    color: #e8e4dc !important;
}

/* Header */
.header-block {
    text-align: center;
    padding: 2.4rem 1rem 1rem;
    border-bottom: 1px solid #2a2a2e;
    margin-bottom: 1.6rem;
}
.header-block h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: #f5eedf !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem !important;
}
.header-block p {
    color: #7c7971 !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
}

/* Panels */
.panel-label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #7c7971 !important;
    margin-bottom: 0.5rem !important;
}

/* Image boxes */
.image-box {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #2a2a2e !important;
    background: #161618 !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #c9a96e !important;
}

/* Tone badge */
.tone-badge {
    display: inline-block;
    background: #1e1c19;
    border: 1px solid #c9a96e44;
    color: #c9a96e;
    border-radius: 6px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    margin-top: 0.5rem;
}

/* Buttons */
.run-btn {
    background: #c9a96e !important;
    color: #0e0e10 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    transition: background 0.2s !important;
}
.run-btn:hover {
    background: #dfc08a !important;
}

/* Sidebar */
.sidebar {
    background: #13131a !important;
    border-radius: 12px !important;
    border: 1px solid #22222a !important;
    padding: 1.2rem !important;
}

/* Accordion */
.accordion {
    background: #161618 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 8px !important;
}

/* Gradio overrides */
.gradio-container .gr-button-primary {
    background: #c9a96e !important;
    border-color: #c9a96e !important;
    color: #0e0e10 !important;
}
label span {
    color: #a09890 !important;
    font-size: 0.82rem !important;
}
.gr-block.gr-box, .gr-form {
    background: #161618 !important;
    border-color: #2a2a2e !important;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Portrait Skin Enhancement") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-block">
        <h1>Portrait Skin Enhancement</h1>
        <p>Classical image-processing pipeline &nbsp;·&nbsp; Edge-preserving smoothing &nbsp;·&nbsp; Tone normalisation</p>
    </div>
    """)

    # ── Main layout: left sidebar + right preview ────────────────────────────
    with gr.Row(equal_height=False):

        # ── LEFT: Controls ───────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280, elem_classes="sidebar"):

            gr.HTML('<p class="panel-label">Input Photo</p>')
            input_img = gr.Image(
                label="",
                type="numpy",
                sources=["upload", "webcam"],
                elem_classes="image-box",
            )

            gr.HTML('<p class="panel-label" style="margin-top:1.2rem">Parameters</p>')

            auto_toggle = gr.Checkbox(
                label="Auto-tune parameters (recommended)",
                value=True,
            )

            with gr.Group(visible=False) as manual_group:
                radius_sl = gr.Slider(
                    minimum=1, maximum=10, step=1, value=3,
                    label="Smoothing Radius  (Guided Filter r)",
                    info="Higher = stronger smoothing"
                )
                strength_sl = gr.Slider(
                    minimum=0.1, maximum=1.0, step=0.05, value=0.5,
                    label="Blend Strength",
                    info="0 = original, 1 = fully smoothed"
                )

            brightness_sl = gr.Slider(
                minimum=0.8, maximum=2.0, step=0.05, value=1.5,
                label="Brightness Boost",
                info="1.0 = neutral, 1.5 = standard lift, 2.0 = strong"
            )

            # Show/hide manual group based on toggle
            auto_toggle.change(
                fn=lambda v: gr.update(visible=not v),
                inputs=auto_toggle,
                outputs=manual_group
            )

            run_btn = gr.Button("✦  Enhance Portrait", elem_classes="run-btn", variant="primary")

            tone_out = gr.Textbox(
                label="", placeholder="Skin tone will appear here after processing…",
                interactive=False, lines=1
            )

        # ── RIGHT: Before / After ────────────────────────────────────────────
        with gr.Column(scale=2):

            gr.HTML('<p class="panel-label">Enhanced Result</p>')

            after_img = gr.Image(
                label="", interactive=False,
                elem_classes="image-box"
            )

            gr.HTML('<p class="panel-label" style="margin-top:1.2rem">Download</p>')
            download_btn = gr.File(label="", interactive=False)

    # ── Wire up the run button ───────────────────────────────────────────────
    def process(img, radius, strength, brightness, auto):
        result_rgb, out_path, tone_label = run_pipeline(
            img, radius, strength, brightness, auto
        )
        return result_rgb, out_path, tone_label

    run_btn.click(
        fn=process,
        inputs=[input_img, radius_sl, strength_sl, brightness_sl, auto_toggle],
        outputs=[after_img, download_btn, tone_out],
    )

    # ── How-it-works accordion ───────────────────────────────────────────────
    with gr.Accordion("How it works", open=False, elem_classes="accordion"):
        gr.Markdown("""
| Step | Module | Transform |
|------|--------|-----------|
| 1 | loader.py | BGR → YCrCb + Grayscale color space transforms |
| 2 | detector.py | HOG + SVM face detection (Dlib), Haar fallback |
| 3 | landmarks.py | 68-point shape regression (Dlib ERT model) |
| 4 | skin_tone.py | YCrCb luminance sampling → Light / Medium / Dark |
| 5 | mask.py | Convex hull mask + Gaussian feathering |
| 6 | smoother.py | Guided Filter (L2-regularised edge-preserving) |
| 7 | tone.py | Histogram stretching on Y channel |
| 8 | blender.py | Alpha compositing (Porter-Duff over) |
        """)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on local network
        server_port=7860,
        share=False,             # set True to get a public gradio.live link
        show_error=True,
    )