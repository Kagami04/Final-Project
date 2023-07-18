import cv2
import numpy as np
import os

# Face detection is done
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        r'C:\Users\HENDRIX\PycharmProjects\face_detection\haarcascade_frontalface_alt.xml')  # Give path to haar classifier as i have given
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)
    return faces, gray_img


# Labels for training data has been created

def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("img_path", img_path)
            print("id: ", id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Not Loaded Properly")
                continue

            faces_rect, gray_img = faceDetection(test_img)
            if len(faces_rect) != 1:
                continue
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y + w, x:x + h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


# Here training Classifier is called
def train_classifier(face,faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face, np.array(faceID))
    return face_recognizer


# Drawing a Rectangle on the Face Function
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 0), thickness= 3)


# Putting text on images
def put_text_with_background(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

