from src.loader import load_image
from src.detector import detect_main_face

bgr, gray, ycrcb = load_image('input/test.jpg')
x, y, w, h = detect_main_face(gray)
print('Face box — x:', x, ' y:', y, ' w:', w, ' h:', h)
