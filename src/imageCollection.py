import numpy as np
import cv2 as cv
import os
import picamera
import time

CWD = '/home/pi/Desktop/ChessMate/'
img_folder = CWD + 'data/image_collection/'


def data_Collection():

	i = 0
	prefix = 'img_'
	suffix = '.PNG' 

	with picamera.PiCamera() as camera:
		camera.start_preview()
		time.sleep(2)

		filename = img_folder + prefix + str(i) + suffix
		camera.capture( filename, resize=(1920,1080) )



if __name__ == "__main__":
	data_Collection()