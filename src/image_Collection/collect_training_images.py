import os
import pickle
import cv2 as cv
import numpy as np
import picamera
import picamera.array
import time

CWD = '/home/pi/Desktop/ChessMate/'
temp_image_folder = CWD + 'data/train/temp/'


def roi_filter(img):

	rows, cols = img.shape[:2]


	y1 = int( rows * 0.055 )
	y2 = int( rows * 0.80 )

	x1 = int( cols * 0.305 )
	x2 = int( cols * 0.72 )

	img = img[ y1:y2, x1:x2]
	return img

def img_pipeline(img):
	img = roi_filter(img)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

	return img


def store_image(img, image_folder, prefix, suffix, zeros = 5):

	if( os.path.isdir( image_folder ) == True ):

		sorted_list = os.listdir( image_folder )
		sorted_list.sort()
		
		i = len(sorted_list)
		num = str(i).zfill(zeros)
		
		filename = image_folder + prefix + num + suffix
		cv.imwrite(filename, img)


def take_raw_image():
	h = 1080
	w = 1920
	i = 0
	prefix = 'temp_image_'
	suffix = '.JPG'


	with picamera.PiCamera() as camera:
		camera.start_preview()
		time.sleep(2)
		
		camera.resolution = (w, h)

		while(i < 1):

			with picamera.array.PiRGBArray(camera) as stream:

				camera.capture( stream, format='rgb' )

				output = stream.array

				image = img_pipeline(output)

				store_image(image, temp_image_folder, prefix, suffix)

				
			i+=1
				
	camera.close()


if __name__ == "__main__":
	take_raw_image()