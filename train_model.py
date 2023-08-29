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

option 3  firebase 

import numpy as np
import cv2
import os
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase
cred = credentials.Certificate("./key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'images-23bb4.appspot.com'
}, name='storage')

import face_recognition as fr

print(fr)


# Function to crop the face region using Haar Cascade
def crop_face(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = gray[y:y + h, x:x + w]
        return cropped_face
    return None


# Function to apply data augmentation to the images
def apply_data_augmentation(image):
    # Random rotation (angle in degrees)
    angle = np.random.randint(-20, 20)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
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


# Function to read and preprocess images for training with data augmentation from Firebase Storage
# Function to read and preprocess images for training with data augmentation from Firebase Storage
def preprocess_images_from_firebase(bucket_name):
    faces = []
    faceID = []

    bucket = storage.bucket(app=firebase_admin.get_app(name='storage'))
    blobs = bucket.list_blobs(prefix=bucket_name)

    for blob in blobs:
        if blob.name.endswith(".jpg") or blob.name.endswith(".png"):
            img_path = blob.name
            print("img_path", img_path)
            # Extract the folder (label) name from the blob path
            folder_name = os.path.dirname(img_path).split('/')[-1]  # Get the last part after splitting by '/'
            print("id: ", folder_name)

            # Read image data from Firebase Storage and convert to OpenCV format
            image_data = blob.download_as_bytes()
            img_np = np.frombuffer(image_data, dtype=np.uint8)
            test_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if test_img is None:
                print("Not Loaded Properly")
                continue

            cropped_face = crop_face(test_img)
            if cropped_face is None:
                print("No face detected in the image:", img_path)
                continue

            rotated_face, scaled_face, flipped_face = apply_data_augmentation(cropped_face)

            faces.extend([cropped_face, rotated_face, scaled_face, flipped_face])
            faceID.extend([int(folder_name)] * 4)

    return faces, faceID


# Training will begin from here
train_folder_path = 'images'  # Path to the folder structure on Firebase Storage
faces, faceID = preprocess_images_from_firebase(train_folder_path)

for img_path, id in zip(faceID, faceID):
    print("Image Path:", img_path)
    print("ID:", id)

# Create the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Train the classifier
face_recognizer.train(faces, np.array(faceID))

# Save the trained model
face_recognizer.save('firebase.xml')

name = {
    0: " ",
    1: "Justine Ballelos",
    2: "Hendrix Pascua",
}

# ... Rest of your code ...


option 4 naka hiwalay na name


import numpy as np
import cv2
import os
import firebase_admin
from firebase_admin import credentials, storage
import json
import face_recognition as fr
print(fr)

with open('name_mapping.json', 'r') as f:
    name_mapping = json.load(f)

# Initialize Firebase
cred = credentials.Certificate("./key.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'images-23bb4.appspot.com'
}, name='storage')

# Function to crop the face region using Haar Cascade
def crop_face(image):
    face_cascade_path = './haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = gray[y:y + h, x:x + w]
        return cropped_face
    return None

# Function to apply data augmentation to the images
def apply_data_augmentation(image):
    angle = np.random.randint(-20, 20)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    scale = np.random.uniform(0.8, 1.2)
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)

    if np.random.rand() < 0.5:
        flipped_image = cv2.flip(image, 1)
    else:
        flipped_image = image

    # Add brightness adjustment
    brightness_factor = np.random.uniform(0.5, 1.5)
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Add slight distortion
    distortion_factor = np.random.uniform(-0.05, 0.05)
    distorted_image = image.copy()
    for i in range(rows):
        distorted_image[i] = np.roll(distorted_image[i], int(cols * distortion_factor))

    return rotated_image, scaled_image, flipped_image, brightened_image, distorted_image

# Function to read and preprocess images for training with data augmentation from Firebase Storage
def preprocess_images_from_firebase(bucket_name, save_folder):
    faces = []
    ids = []

    bucket = storage.bucket(app=firebase_admin.get_app(name='storage'))
    blobs = bucket.list_blobs(prefix=bucket_name)

    for blob in blobs:
        if blob.name.endswith(".jpg") or blob.name.endswith(".png"):
            img_path = blob.name
            print("img_path", img_path)
            folder_name = os.path.dirname(img_path).split('/')[-1]
            print("id: ", folder_name)

            image_data = blob.download_as_bytes()
            img_np = np.frombuffer(image_data, dtype=np.uint8)
            test_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if test_img is None:
                print("Not Loaded Properly")
                continue

            cropped_face = crop_face(test_img)
            if cropped_face is None:
                print("No face detected in the image:", img_path)
                continue

            augmented_images = apply_data_augmentation(cropped_face)
            for idx, augmented_image in enumerate(augmented_images):
                save_path = os.path.join(save_folder, f"{folder_name}_{idx}.jpg")
                cv2.imwrite(save_path, augmented_image)
                faces.append(augmented_image)
                ids.append(int(folder_name))

    return faces, ids

# Function to train the classifier using LBPHFaceRecognizer
def lbph_classifier(faces, ids):
    lbph_classifier = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    lbph_classifier.train(faces, np.array(ids))
    return lbph_classifier

# Training will begin from here
train_folder_path = 'images'
save_folder = 'abay_ho'
faces, ids = preprocess_images_from_firebase(train_folder_path, save_folder)

# Train the classifier with data augmentation
augmented_faces = []
augmented_ids = []
for i in range(len(faces)):
    face = faces[i]
    label = ids[i]
    rotated_face, scaled_face, flipped_face, brightened_face, distorted_face = apply_data_augmentation(face)
    augmented_faces.extend([face, rotated_face, scaled_face, flipped_face, brightened_face, distorted_face])
    augmented_ids.extend([label] * 6)

# Train the classifier using the augmented data
lbph_recognizer = lbph_classifier(augmented_faces, augmented_ids)

# Save the trained model
lbph_recognizer.save('firebase_augmented_mapping.xml')



# ... (rest of the code)

option 5 nadagdagan ng accuracy

import numpy as np
import cv2
import os
import firebase_admin
from firebase_admin import credentials, storage
import json
import face_recognition as fr

print(fr)

with open('name_mapping.json', 'r') as f:
    name_mapping = json.load(f)

# Determine whether you want to use Firebase or not
want_to_use_firebase = True  # Set to True if you want to use Firebase, False otherwise

# Initialize Firebase
if want_to_use_firebase:
    cred = credentials.Certificate('./key.json')
    firebase_admin.initialize_app(cred, {'storageBucket': 'images-23bb4.appspot.com'}, name='storage')

# Function to read and preprocess images for training with data augmentation from Firebase Storage
def labels_for_training_data(bucket_name, save_folder):
    faces = []
    ids = []

    bucket = storage.bucket(app=firebase_admin.get_app(name='storage'))
    blobs = bucket.list_blobs(prefix=bucket_name)

    for blob in blobs:
        if blob.name.endswith(".jpg") or blob.name.endswith(".png"):
            img_path = blob.name
            print("img_path", img_path)
            folder_name = os.path.dirname(img_path).split('/')[-1]
            print("id: ", folder_name)

            image_data = blob.download_as_bytes()
            img_np = np.frombuffer(image_data, dtype=np.uint8)
            test_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if test_img is None:
                print("Not Loaded Properly")
                continue

            cropped_face = crop_face(test_img)
            if cropped_face is None:
                print("No face detected in the image:", img_path)
                continue

            augmented_images = apply_data_augmentation(cropped_face)
            for idx, augmented_image in enumerate(augmented_images):
                save_path = os.path.join(save_folder, f"{folder_name}_{idx}.jpg")
                cv2.imwrite(save_path, augmented_image)
                faces.append(augmented_image)
                ids.append(int(folder_name))

    return faces, ids

# Function to crop the face region using Haar Cascade
def crop_face(image):
    cascade_path = './haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = gray[y:y + h, x:x + w]
        return cropped_face
    return None



# Function to apply data augmentation to the images
def apply_data_augmentation(image):
    angle = np.random.randint(-20, 20)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Resize to a smaller size for far detection
    smaller_size = np.random.randint(40, 80)
    smaller_image = cv2.resize(image, (smaller_size, smaller_size))

    if np.random.rand() < 0.5:
        flipped_image = cv2.flip(image, 1)
    else:
        flipped_image = image

    # Add brightness adjustment
    brightness_factor = np.random.uniform(0.5, 0.8)
    darkened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Add Gaussian noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply salt and pepper noise
    s_vs_p = 0.5
    amount = 0.04
    noisy_sp_image = image.copy()
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_sp_image[coords[0], coords[1]] = 255

    return [
        rotated_image,
        smaller_image,
        flipped_image,
        darkened_image,
        noisy_image,
        blurred_image,
        noisy_sp_image,
    ]

# Function to apply data augmentation to the images at multiple scales

# Training will begin from here
train_folder_path = 'images'
save_folder = 'abay_ho'

# Preprocess images with data augmentation
faces, ids = labels_for_training_data(train_folder_path, save_folder)

# Train the classifier with data augmentation
augmented_faces = []
augmented_ids = []

for i in range(len(faces)):
    face = faces[i]
    label = ids[i]
    augmented_images = apply_data_augmentation(face)

    augmented_faces.extend(augmented_images)
    augmented_ids.extend([label] * len(augmented_images))

# Train the classifier using augmented data
recognizer = fr.train_classifier(faces, np.array(ids))

# Save the trained model
recognizer.write('firebase.yml')

print("Training Complete")
