import cv2
import os
import datetime
import face_recognition as fr

print(fr)

# Define the parameters to tune
radius = 1
neighbors = 8
grid_x = 8
grid_y = 8

# Create the face recognizer with the desired parameters
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
face_recognizer.read(r'C:\Users\HENDRIX\PycharmProjects\pythonProject1\last.xml')

cap = cv2.VideoCapture(0)  # If you want to recognize faces from a video, replace 0 with the video path
known_faces_folder = r'C:\Users\HENDRIX\PycharmProjects\pythonProject1\known_faces'
unknown_faces_folder = r'C:\Users\HENDRIX\PycharmProjects\pythonProject1\unknown_faces'

name = {
    0: "Justine Ballelos",
    1: "Joy Roldan",
    2: "Hendrix Pascua"
}

threshold = 120

# Create folders if they don't exist
if not os.path.exists(known_faces_folder):
    os.makedirs(known_faces_folder)
if not os.path.exists(unknown_faces_folder):
    os.makedirs(unknown_faces_folder)

# Dictionary to keep track of snapshots taken for each label
snapshots_taken = {label: 0 for label in name.keys()}

# Counter to keep track of "Not Registered" faces
not_registered_counter = 0

def save_snapshot(image, folder_path, file_name):
    # Save the image as a snapshot
    snapshot_path = os.path.join(folder_path, file_name)
    cv2.imwrite(snapshot_path, image)
    print('Snapshot saved at:', snapshot_path)

def take_snapshot(face_image, folder_path, label):
    global snapshots_taken, not_registered_counter
    if label in name:
        if snapshots_taken[label] == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f'snapshot_{name[label]}_{timestamp}.jpg'
            snapshot_path = os.path.join(folder_path, file_name)
            cv2.imwrite(snapshot_path, face_image)
            print('Snapshot saved at:', snapshot_path)
            snapshots_taken[label] += 1
    else:
        not_registered_counter += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f'snapshot_NotRegistered_{not_registered_counter}_{timestamp}.jpg'
        snapshot_path = os.path.join(folder_path, file_name)
        cv2.imwrite(snapshot_path, face_image)
        print('Snapshot saved at:', snapshot_path)

# Get the height of the frame to draw the line slightly below the center
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_y = frame_height // 3  # Move the line higher by decreasing the value

# Flag variable to track whether the line has been crossed by a face
line_crossed = False

while True:
    ret, test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("Faces Detected: ", faces_detected)

    # Draw a horizontal line where face detection and recognition start
    cv2.line(test_img, (0, line_y), (test_img.shape[1], line_y), (0, 0, 255), 2)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 0), thickness=3)

        # Check if the face is above the line and the top of the face is above the line
        if y < line_y:
            continue  # Skip the face recognition for faces below the line

        roi_gray = gray_img[y:y + h, x:x + w]

        # Save the original size face region as the snapshot
        face_image = gray_img[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence:", confidence)
        print("Label:", label)
        fr.draw_rect(test_img, (x, y, w, h))

        if confidence > threshold:
            predicted_name = name[label]
            take_snapshot(face_image, known_faces_folder, label)
        else:
            predicted_name = "Not Registered"
            take_snapshot(face_image, unknown_faces_folder, -1)

        fr.put_text_with_background(test_img, predicted_name, x, y)

        # Set the flag to True to indicate the line has been crossed
        line_crossed = True

    # Reset the flag if there are no faces detected above the line
    if len(faces_detected) == 0:
        line_crossed = False

    resized_img = cv2.resize(test_img, (1000, 700))

    cv2.imshow("Face Detection", resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



option 2 firebase save ss

import cv2
import datetime
import face_recognition as fr
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("./key.json")  # Path to your Firebase Admin SDK credentials
firebase_admin.initialize_app(cred, {'storageBucket': 'images-23bb4.appspot.com'})
bucket = storage.bucket()

print(fr)

# Define the parameters to tune
radius = 1
neighbors = 8
grid_x = 8
grid_y = 8

# Create the face recognizer with the desired parameters
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
face_recognizer.read(r'C:\Users\HENDRIX\PycharmProjects\pythonProject1\firebase.xml')

cap = cv2.VideoCapture(0)  # If you want to recognize faces from a video, replace 0 with the video path

name = {
    0: " ",
    1: "Justine Ballelos",
    2: "Hendrix Pascua",
}

known_faces_folder = "known_faces"
unknown_faces_folder = "unknown_faces"

threshold = 120

# Dictionary to keep track of snapshots taken for each label
snapshots_taken = {label: 0 for label in name.keys()}

# Counter to keep track of "Not Registered" faces
not_registered_counter = 0

def save_snapshot(image, folder_path, file_name):
    blob = bucket.blob(f'{folder_path}/{file_name}')
    blob.upload_from_filename(image)
    print('Snapshot uploaded to Firebase:', file_name)

def take_snapshot(face_image, folder_path, label):
    global snapshots_taken, not_registered_counter
    if label in name:
        if snapshots_taken[label] == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f'snapshot_{name[label]}_{timestamp}.jpg'
            cv2.imwrite(file_name, face_image)
            save_snapshot(file_name, folder_path, file_name)
            snapshots_taken[label] += 1
    else:
        not_registered_counter += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f'snapshot_Not_Registered_{not_registered_counter}_{timestamp}.jpg'
        cv2.imwrite(file_name, face_image)
        save_snapshot(file_name, folder_path, file_name)


# Get the height of the frame to draw the line slightly below the center
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_y = frame_height // 3  # Move the line higher by decreasing the value

# Flag variable to track whether the line has been crossed by a face
line_crossed = False

while True:
    ret, test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("Faces Detected: ", faces_detected)

    # Draw a horizontal line where face detection and recognition start
    cv2.line(test_img, (0, line_y), (test_img.shape[1], line_y), (0, 0, 255), 2)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 0), thickness=3)

        # Check if the face is above the line and the top of the face is above the line
        if y < line_y:
            continue  # Skip the face recognition for faces below the line

        roi_gray = gray_img[y:y + h, x:x + w]

        # Save the original size face region as the snapshot
        face_image = gray_img[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence:", confidence)
        print("Label:", label)
        fr.draw_rect(test_img, (x, y, w, h))


        if confidence > threshold:
            predicted_name = name[label]
            take_snapshot(face_image, known_faces_folder, label)
        else:
            predicted_name = "Not Registered"
            take_snapshot(face_image, unknown_faces_folder, -1)

        fr.put_text_with_background(test_img, predicted_name, x, y)

        # Set the flag to True to indicate the line has been crossed
        line_crossed = True

    # Reset the flag if there are no faces detected above the line
    if len(faces_detected) == 0:
        line_crossed = False

    resized_img = cv2.resize(test_img, (1000, 700))

    cv2.imshow("Face Detection", resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
