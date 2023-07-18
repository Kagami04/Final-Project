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



