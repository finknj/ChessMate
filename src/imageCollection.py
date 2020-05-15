import numpy as np
import cv2 as cv
import os
import picamera
import time

CWD = '/home/pi/Desktop/ChessMate/'
img_folder = CWD + 'data/image_collection/'


def roi_filter(img):

	rows, cols = img.shape[:2]


	y1 = int( rows * 0.07 )
	y2 = int( rows * 0.88 )

	x1 = int( cols * 0.31 )
	x2 = int( cols * 0.89 )

	img = img[ y1:y2, x1:x2]
	return img


def img_pipeline(img):

	img = roi_filter(img)

	return img

def data_Collection():
	h = 1080
	w = 1920
	i = 0
	prefix = 'img_'
	suffix = '.JPG' 

	with picamera.PiCamera() as camera:
		camera.resolution = (w, h)
		while(i < 1):

			output = np.empty((1088*1920*3), dtype = np.uint8)
			camera.capture( output, 'rgb' )

			output = output.reshape( (1088, 1920, 3) )
			output = output[:w, :h, :]

			image = img_pipeline(output)
			filename = img_folder + prefix + str(i) + suffix
			
			cv.imwrite(filename, image)
			
			i+=1
			



if __name__ == "__main__":
	data_Collection()
