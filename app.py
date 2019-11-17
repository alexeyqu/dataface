# import the necessary packages
from __future__ import print_function
import datetime
import imutils
from imutils.video import VideoStream
from PIL import Image
import cv2
import os

from cv.recognition import FacialRecognizer

import sys
import time
import numpy as np

import cv2
import tkinter
from tkinter import simpledialog


video_stream_path = 'http://192.168.1.109:4747/video'
recognizer = FacialRecognizer('db.sqlite')

clicked = False
mouseX, mouseY = None, None

root = tkinter.Tk()
root.withdraw()

def add_new_person(img):
    name = simpledialog.askstring('Who is this?', 'Enter the name')
    recognizer.assign_name_to_image(np.asarray(img), name)

def clck_handler(event, x, y, flags, param):
    global mouseX, mouseY, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY, clicked = x, y, True
        print(mouseX, mouseY, clicked)


cv2.namedWindow('video')
cv2.setMouseCallback('video', clck_handler)


class FaceApp:
    def __init__(self, vs):
        self.vs = vs

    def run(self):
        global clicked, mouseX, mouseY
        process_ratio = 5
        process = 0
        faces = []
        frame = None
        try:
            while True:
                frame = self.vs.read()
                if process % process_ratio == 0:
                    process = 0
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    imagearr = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    faces = recognizer.recognize_faces(imagearr)

                    for face in faces:
                        cv2.rectangle(frame, (face.left * 4, face.top * 4), (face.right * 4, face.bottom * 4), (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, face.name or 'unknown', (face.left * 4 + 6, face.bottom * 4 - 6), font, 1.0, (255, 255, 255), 1)
                    cv2.imshow('video', frame)

                process += 1

                if clicked:
                    clicked = False
                    for face in faces:
                        if face.top < mouseY * 0.25 < face.bottom and face.left < mouseX * 0.25 < face.right and face.name is None:
                            print(face)
                            cropped = Image.fromarray(imagearr).crop((face.left, face.top, face.right, face.bottom))
                            cropped.save('test.jpg')
                            w, h = cropped.size
                            add_new_person(cropped.resize((w // 4, h // 4)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    del recognizer
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
