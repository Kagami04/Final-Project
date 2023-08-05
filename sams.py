import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2

cred = credentials.Certificate("./photos.json")
app = firebase_admin.initialize_app(cred, {
    'storageBucket': 'photos-29104.appspot.com'  # Replace with your Firebase storage bucket name
})

bucket = storage.bucket()
blob = bucket.get_blob("asseh.jpg") #blob
arr = np.frombuffer(blob.download_as_string(), np.uint8) #array of bytes
img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555) #actual image

cv2.imshow('image', img)
cv2.waitKey(0)