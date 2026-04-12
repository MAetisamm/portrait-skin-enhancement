# Portrait Skin Enhancement

A traditional image-processing pipeline that enhances facial skin
in portrait photos while preserving eyes, eyebrows, lips, and hair.

## Pipeline Steps
1. Load image & convert color spaces
2. Face detection (Haar Cascade)
3. Facial landmark detection (Dlib 68-point)
4. Skin mask construction (YCrCb thresholding)
5. Skin smoothing (Guided Filter)
6. Tone enhancement (CLAHE)
7. Alpha blending & save

## Technologies
- Python 3.11
- OpenCV + OpenCV-contrib
- Dlib
- NumPy

## Setup

### 1. Clone the repo
git clone https://github.com/yourusername/portrait-skin-enhancement.git
cd portrait-skin-enhancement

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt