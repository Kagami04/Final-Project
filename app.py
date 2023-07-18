from flask import Flask, send_from_directory, jsonify, render_template, Response
import cv2
import os
import datetime
import face_recognition as fr
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('shorties.yml')

cap = cv2.VideoCapture(0)  # If you want to recognize faces from a video, replace 0 with the video path

known_faces_folder = 'known_faces'
unknown_faces_folder = 'unknown_faces'

name = {
    0: "Not Registered",
    1: "Justine Ballelos",
    2: "Mark Hendrix",
    3: "Joy Roldan"
}

threshold = 90

if not os.path.exists(known_faces_folder):
    os.makedirs(known_faces_folder)
if not os.path.exists(unknown_faces_folder):
    os.makedirs(unknown_faces_folder)

snapshot_taken = {label: False for label in name.keys()}

def save_snapshot(image, folder_path, file_name):
    snapshot_path = os.path.join(folder_path, file_name)
    cv2.imwrite(snapshot_path, image)
    print('Snapshot saved at:', snapshot_path)

def take_snapshot(face_image, folder_path, label):
    global snapshot_taken
    if not snapshot_taken[label]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_snapshot(face_image, folder_path, f'snapshot_{name[label]}_{timestamp}.jpg')
        snapshot_taken[label] = True

def detect_faces():
    global latest_snapshot

    while True:
        ret, test_img = cap.read()
        faces_detected, gray_img = fr.faceDetection(test_img)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 0), thickness=3)

        for face in faces_detected:
            (x, y, w, h) = face
            roi_gray = gray_img[y:y + h, x:x + w]

            face_image = gray_img[y:y + h, x:x + w]

            label, confidence = face_recognizer.predict(roi_gray)
            fr.draw_rect(test_img, face)

            if confidence < threshold:
                predicted_name = name[label]
                take_snapshot(face_image, known_faces_folder, label)
            else:
                predicted_name = "Not Registered"
                take_snapshot(face_image, unknown_faces_folder, 0)

            fr.put_text_with_background(test_img, predicted_name, x, y)

            latest_snapshot = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'label': predicted_name,
                'image_data': encode_image(test_img)
            }

        resized_img = cv2.resize(test_img, (1000, 700))
        _, jpeg = cv2.imencode('.jpg', resized_img)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshots')
def get_snapshots():
    snapshots = []
    for folder_path, label in [(known_faces_folder, 1), (unknown_faces_folder, 0)]:
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                if folder_path == known_faces_folder:
                    # For known faces, the filename contains the person's name
                    label_name = filename.split("_")[1]
                    timestamp_str = datetime.datetime.now().strftime("%H:%M %d-%m-%Y")
                else:
                    # For unknown faces, use the current time as the timestamp
                    label_name = name[label]
                    timestamp_str = datetime.datetime.now().strftime("%H:%M %d-%m-%Y")

                with open(os.path.join(folder_path, filename), "rb") as file:
                    image_data = base64.b64encode(file.read()).decode("utf-8")

                snapshots.append({
                    "timestamp": timestamp_str,
                    "label": label_name,
                    "image_data": image_data
                })

    return jsonify(snapshots)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
