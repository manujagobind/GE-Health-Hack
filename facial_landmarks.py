#!/usr/bin/env pythonn
"""
= = = = =    EMOTION RECOGNITION    = = = = =

This module is uses DLIB to extract facial landmarks (68 in total)
and cretes a feature vector using these landmarks.

These landmarks can be used to examine a person's emotions
using his/her facial expressions.
A particualr expression will always have similar landmark positions.

We have then created a classifier using SVM for the feature vectors
in order to predict expressions.

More training data is required for improving the efficieny.

- - - - -
To start press ENTER
To stop press Q
"""
import os, math, pickle, argparse, cv2, dlib, warnings
from sklearn import svm
from tqdm import tqdm
import numpy as np

class FacialLandmarks(object):

    def __init__(self):

        self.images_path = os.getcwd() + "/data/images/emotions/"
        self.image_filenames = os.listdir(self.images_path)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        self.classifier = svm.SVC(kernel='linear', probability=True, tol=1e-3)

        self.X, self.Y = [], []

    def get_landmarks(self, image):
        try:
            detections = self.detector(image, 1)
            for k, d in enumerate(detections):
                shape = self.predictor(image, d)
                xList, yList =  [], []
                for i in range(1, 68):
                    xList.append(float(shape.part(i).x))
                    yList.append(float(shape.part(i).y))

            #Find centre of gravity
            xMean = np.mean(xList)
            yMean = np.mean(yList)

            #Calculate distance of each point from centre of gravity
            #i.e. Relative Distance
            xCentral = [(x-xMean) for x in xList]
            yCentral = [(y-yMean) for y in yList]

            #Calculate nose angle
            if xList[26] == xList[29]:
                nose_angle = 0
            else:
                nose_angle = int(math.atan( (yList[26]-yList[29]) / (xList[26] - xList[29]) ) * 180 / math.pi )

            if nose_angle < 0:
                nose_angle += 90
            else:
                nose_angle -= 90

            landmarks_vectorised = []

            for x, y, z, w in zip(xCentral, yCentral, xList, yList):
                #Append distance of each point relative to centre of gravity
                landmarks_vectorised.append(x)
                landmarks_vectorised.append(y)
                #Calculate Eucledian Distance b/w each point and the centre point
                meannp = np.asarray((yMean, xMean))
                coornp = np.asarray((z, w))
                dist = np.linalg.norm(coornp - meannp)

                angle_relative = ( math.atan( (z-yMean) / (w-xMean) ) * 180/math.pi ) - nose_angle
                landmarks_vectorised.append(angle_relative)

                #For cases when no face is detected
            if not len(detections):
                landmarks_vectorised = None
            return landmarks_vectorised
        except:
            print "[ OPPS ] An exception occured while generating facial landmarks."

    def get_class(self, filename):
        #Depends on the dataset being used for training
        key = filename[4:6]
        if key == "AF":
            return 0
        elif key == "AN":
            return 1
        elif key == "DI":
            return 2
        elif key == "HA":
            return 3
        elif key == "NE":
            return 4
        elif key == "SA":
            return 5
        elif key == "SA":
            return 6
        else:
            return 7

    def get_emotion(self, index):
        #Depends on the dataset being used for training
        if index==0:
            print "[ EMOTION RECOGNIZED ] Afraid"
        elif index==1:
            print "[ EMOTION RECOGNIZED ] Angry"
        elif index==2:
            print "[ EMOTION RECOGNIZED ] Disgust"
        elif index==3:
            print "[ EMOTION RECOGNIZED ] Happy"
        elif index==4:
            print "[ EMOTION RECOGNIZED ] Neutral"
        elif index==5:
            print "[ EMOTION RECOGNIZED ] Sad"
        elif index==6:
            print "[ EMOTION RECOGNIZED ] Surprised"
        else:
            print "[ EMOTION RECOGNIZED ] Error"

    def extract_features(self):
        try:
            print "[ TASK ] Extracting features."
            num_total_images = len(self.image_filenames)
            for i in tqdm(range(num_total_images)):
                landmarks_vector = self.get_landmarks(self.clahe.apply(cv2.imread(self.images_path + self.image_filenames[i], 0)))
                if landmarks_vector:
                    #Append training examples
                    self.X.append(landmarks_vector)
                    self.Y.append(self.get_class(self.image_filenames[i]))
                #Stopping early 'causeo of an error exception that wasn't understood
                if i == 350:
                    break
            self.X = np.array(self.X)
            self.Y = np.array(self.Y)
            print "[ DONE ] Feature extraction complete."
            print "[ TASK ] Saving features for future use."
            with open('dynamic/facial_landmarks.pkl', 'wb') as fl:
                pickle.dump(self.X, fl)
                pickle.dump(self.Y, fl)
            print "[ DONE ] Features saved."
        except:
            print "[ OPPS ] An exception occured while extracting features."

    def train_classifier(self):
        try:
            print "[ TASK ] Loading saved features."
            with open('dynamic/facial_landmarks.pkl', 'rb') as fl:
                self.X = pickle.load(fl)
                self.Y = pickle.load(fl)
            print "[ DONE ] Features loaded."
            print "[ TASK ] Training classifier."
            self.classifier.fit(self.X, self.Y)
            print "[ DONE ] Training complete."
            print "[ TASK ] Saving current state."
            with open('dynamic/facial_landmarks_clf.pkl', 'wb') as flc:
                pickle.dump(self.classifier, flc)
            print "[ TASK ] Save complete."
        except:
            print "[ OPPS ] An excepton occured while training classifier."

    def recognize_emotions(self):
        try:
            with open('dynamic/facial_landmarks_clf.pkl', 'rb') as flc:
                self.classifier = pickle.load(flc)
            cap = cv2.VideoCapture(0)
            frame_count = 0
            while cap.isOpened():
                _, frame = cap.read()
                cv2.imshow('FL_Frame', frame)
                if frame_count % 25 == 0:
                    clahe_img = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    self.get_emotion(self.classifier.predict(np.array(self.get_landmarks(clahe_img))))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print "[ STOP ] Ending emotion recognition."
                        break
                    frame_count = 0
                frame_count += 1
            cv2.destroyAllWindows()
        except:
            print "[ OPPS ] An exception occured while recognizing emotions."


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print __doc__
    raw_input()
    facial = FacialLandmarks()
    facial.extract_features()
    facial.train_classifier()
    facial.recognize_emotions()
