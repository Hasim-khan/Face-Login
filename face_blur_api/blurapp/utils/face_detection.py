
import cv2
def detect_faces_from_array(bgr_image):
    # bgr_image: numpy array (cv2.imread style)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # return list of (x,y,w,h)
    return faces
