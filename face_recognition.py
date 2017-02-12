#!/usr/bin/env python
"""
= = = = =    FACE RECOGNITION    = = = = = 

This module implements FACE RECOGNITION using LBPH approach.

Few images of patients/doctors need to be saved for training.

Face recognition can help identify the patients as well as the doctors.

- - - - -
Press ENTER to start
Press Q to exit
"""
import cv2, os, csv, pickle
import numpy as np
from PIL import Image

class FaceRecognition(object):

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.createLBPHFaceRecognizer()
        self.images_path = os.getcwd() + '/data/images/faces/'
        self.person_ids = os.listdir(self.images_path)
        self.person_faces, self.person_labels = [], []

    def setup(self):
        print "[ TASK ] Initiate training."
        person_faces, person_labels = [], []
        for person_id in self.person_ids:
            directory = self.images_path + person_id
            for image in os.listdir(directory):
                image_pil = Image.open(directory+"/"+image).convert('L')
                img = np.array(image_pil, 'uint8')
                faces = self.face_cascade.detectMultiScale(img)
                for (x, y, w, h) in faces:
                    self.person_faces.append(img[y:y+h, x:x+w])
                    self.person_labels.append(int(person_id))
        self.recognizer.train(self.person_faces, np.array(self.person_labels))
        print "[ DONE ] Training complete."

    def recognize_faces(self):
        print "[ TASK ] Recognizing."
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, frame = cap.read()
            cv2.imshow('FR_Frames_Live', frame)
            if _:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5 )
                for (x, y, w, h) in faces:
                    predicted_label, confidence = self.recognizer.predict(gray.copy()[y:y+h, x:x+w])
                    print "[ PREDICTION ] Person ID: ", predicted_label
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print __doc__
    raw_input()
    recognizer = FaceRecognition()
    recognizer.setup()
    recognizer.recognize_faces()
