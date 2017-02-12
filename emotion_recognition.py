#!/usr/bin/env python
"""
= = = = =    EMOTION RECOGNITION    = = = = =

This module used Microsoft's cloud based API
for detecting facial emotions in an image.

There is a minimal lag due to request and response
from the cloud.
- - - - -
Press ENTER to start
Press Q to stop
"""
import requests, cv2, time

class EmotionRecognition(object):

    def __init__(self, key):
        """ Configuration and settings. """

        self._url = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'
        self._key = key

        self.headers = dict()
        self.headers['Ocp-Apim-Subscription-Key'] = self._key
        self.headers['Content-Type'] = 'application/octet-stream'

        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        self.maxNumRetries = 10
        self.result = None

    def processRequest(self, data):
        """ Handle API requests. """

        retries = 0
        self.result = None

        response = requests.post(self._url, headers=self.headers, data=data)

        #429: Too many requests
        if response.status_code == 429:
            print "Error: %s" % (response.json()['error']['message'])

            if retries <= self.maxNumRetries:
                time.sleep(1)
                retries += 1
            else:
                print ('Maximum retries limit reached')

        #200 or 201: Success
        elif response.status_code == 200 or response.status_code == 201:

            if 'content-length' in response.headers and not int(response.headers['content-length']):
                self.result = None
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
                if 'application/json' in response.headers['content-type'].lower():
                    self.result = response.json() if response.content else None
                elif 'image' in response.headers['content-type'].lower():
                    self.result = response.content

        else:
            print 'Error Code: %d' % (response.status_code)
            print 'Message: %s' % (response.json()['error']['message'])


    def analyse(self):
        """ Find the emotion with maximum score. """

        for face in self.result:
            scores = face['scores']
            max_score = 0
            emotion = None
            for key in scores:
                if scores[key] > max_score:
                    emotion = key
                    max_score = scores[key]
            print "[ EMOTION RECOGNIZED ] ", emotion
        print


    def start(self):

        while True:
            ret, frame = self.cap.read()
            self.frame_count += 1
            cv2.imshow('Frames', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if self.frame_count % 25  == 0:
                cv2.imwrite('frames.jpg', frame)
                with open('frames.jpg', 'rb') as f:
                    data = f.read()
                self.processRequest(data)
                if self.result:
                    self.analyse()
                self.frame_count = 0

        cv2.destroyAllWindows()

if __name__ == '__main__':
    print __doc__
    raw_input()
    key='ENTER YOUR KEY HERE'
    EmotionRecognition(key=key).start()
