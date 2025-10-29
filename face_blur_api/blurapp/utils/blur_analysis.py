import cv2
import numpy as np

def analyze_blur_from_array(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = variance < 100  # lower threshold → blurrier image
    return is_blurry, round(variance, 3)
