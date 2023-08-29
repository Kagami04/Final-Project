import cv2
import numpy as np


# Face detection is done
def faceDetection(test_img):
    grayscale_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        r'./haarcascade_frontalface_alt.xml')  # Give path to haar classifier as i have given
    faces = cascade.detectMultiScale(grayscale_image,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces, grayscale_image



# Here training Classifier is called

def train_classifier(faces,ids):
    recognizer = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(ids))
    return recognizer

# Drawing a Rectangle on the Face Function
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 0), thickness= 3)


# Putting text on images
def put_text_with_background(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)


