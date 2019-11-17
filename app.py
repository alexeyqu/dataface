# import the necessary packages
from __future__ import print_function
import datetime
import imutils
from imutils.video import VideoStream
import cv2
import os

from cv.recognition import FacialRecognizer

import sys
import time

import cv2


video_stream_path = 'http://192.168.1.109:4747/video'
recognizer = FacialRecognizer('db.sqlite')


class FaceApp:
    def __init__(self, vs):
        self.vs = vs

    def run(self):
        process = True
        try:
            while True:
                frame = self.vs.read()
                if process:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    imagearr = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    faces = recognizer.recognize_faces(imagearr)

                    for face in faces:
                        cv2.rectangle(frame, (face.left * 4, face.top * 4), (face.right * 4, face.bottom * 4), (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, face.name, (face.left + 6, face.bottom - 6), font, 1.0, (255, 255, 255), 1)

                process = not process
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except BaseException as e:
            print(e)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("[INFO] warming up camera...")
    vs = VideoStream(src=video_stream_path).start()
    #vs = VideoStream(usePiCamera=True).start()
    # start the app
    FaceApp(vs).run()
