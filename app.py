# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
from imutils.video import VideoStream
import cv2
import os

from cv.recognition import FacialRecognizer

import sys
import time

from tkinter import ttk
import cv2


video_stream_path = 0#'http://192.168.1.109:4747/video'
recognizer = FacialRecognizer('db.sqlite')


class FaceApp:
	def __init__(self, vs):
		self.vs = vs
		self.frame = None
		self.thread = None
		self.stopEvent = None

		self.root = tki.Tk()
		self.panel = None

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

	def videoLoop(self):
		try:
			while not self.stopEvent.is_set():
				self.frame = self.vs.read()
				# self.frame = imutils.resize(self.frame, width=300)
				imagearr = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(imagearr)
				image = ImageTk.PhotoImage(image)

				print(imagearr.shape, imagearr)

				faces = recognizer.recognize_faces(imagearr)
				# print(imagearr, faces)

				# rects = 
				# for (x, y, w, h) in rects:
				# 	cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tki.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)

					


		            
				    	
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image
					
					
		except BaseException as e:
			print(e)

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()# -*- coding: utf-8 -*-



if __name__ == '__main__':
	# initialize the video stream and allow the camera sensor to warmup
    print("[INFO] warming up camera...")
    vs = VideoStream(src=video_stream_path).start()
    #vs = VideoStream(usePiCamera=True).start()
      
    time.sleep(2.0)
    

	# start the app
    pba = FaceApp(vs)
    pba.root.mainloop()
