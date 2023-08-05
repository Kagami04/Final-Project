import numpy as np
import cv2
import os

import face_recognition as fr
print (fr)


#Training will begin from here

faces,faceID=fr.labels_for_training_data(r'C:\Users\HENDRIX\PycharmProjects\face_detection\images') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\HENDRIX\PycharmProjects\face_detection\Unknown.yml') #It will save the trained model. Just give path to where you want to save

name = {0:" ",
        1:"Justine Ballelos",
        2:"Mark Hendrix",
        3:"Joy Roldan"
       } #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.

option 2 

import numpy as np
import cv2
import os

import face_recognition as fr
print (fr)

# Function to crop the face region using Haar Cascade
def crop_face(image_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = gray[y:y+h, x:x+w]
        return cropped_face
    return None

# Function to apply data augmentation to the images
def apply_data_augmentation(image):
    # Random rotation (angle in degrees)
    angle = np.random.randint(-20, 20)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Random scaling (percentage)
    scale = np.random.uniform(0.8, 1.2)
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)

    # Random horizontal flipping
    if np.random.rand() < 0.5:
        flipped_image = cv2.flip(image, 1)
    else:
        flipped_image = image

    return rotated_image, scaled_image, flipped_image

# Function to read and preprocess images for training with data augmentation
def preprocess_images(directory):
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
            cropped_face = crop_face(img_path)
            if cropped_face is None:
                print("No face detected in the image:", img_path)
                continue

            # Apply data augmentation
            rotated_face, scaled_face, flipped_face = apply_data_augmentation(cropped_face)

            # Add all the augmented faces to the list
            faces.extend([cropped_face, rotated_face, scaled_face, flipped_face])
            faceID.extend([int(id)] * 4)

    return faces, faceID

#Training will begin from here
train_folder_path = r'C:\Users\HENDRIX\PycharmProjects\face_detection\images'
faces, faceID = preprocess_images(train_folder_path)

# Create the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Train the classifier
face_recognizer.train(faces, np.array(faceID))

# Save the trained model
face_recognizer.save(r'C:\Users\HENDRIX\PycharmProjects\face_detection\last.xml')

name = {
    0: " ",
    1: "Justine Ballelos",
    2: "Hendrix Pascua",
    3: "Joy Roldan"
} #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.


