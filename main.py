from src.loader import load_image
from src.detector import detect_main_face
from src.landmarks.py import get_facial_landmarks

bgr, gray, ycrcb = load_image('input/test.jpg')
face_box = detect_main_face(gray)
pts = get_facial_landmarks(bgr, face_box)
print('All done — pts[0]:', pts[0], 'pts[67]:', pts[67])
