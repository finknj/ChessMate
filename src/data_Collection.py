import os
import pickle
import cv2 as cv
import numpy as np
import picamera
import picamera.array
import time

CWD = '/home/pi/Desktop/ChessMate/'
train_image_folder = CWD + 'data/train/'
test_image_path = CWD + 'data/test_images/test_image0.JPG'


picklepath = CWD + 'data/pickle/chessboard_pickle.p'

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

#LOADS THE PREDEFINED DICTIONARY OF SQUARE POSITIONS
def load_dictionary(picklepath = picklepath):
	
	with open(picklepath, mode = 'rb') as f:
		file = pickle.load(f)
		chessboard_dictionary = file['chessboard_pickle']
	return chessboard_dictionary



def parse_full_image(img, chessboard_dictionary):
	
	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	image_list = [[] for i in range(len(chessboard_dictionary))]

	for r in range(0, 8):
		alpha_id = letters[r]
		for c in range(0,8):
			num_id = str(c + 1)
			position_string = alpha_id + num_id
			position = chessboard_dictionary.get(position_string)
			image_list[c + r * 8] = cropimage(img, position)


	return image_list




def cropimage(img, position, buffer = 20):
	
	x1, y1, x2, y2 = position

	y1 = y1 - buffer
	if(y1 < 0): y1 = 0

	#y2 = y2 + buffer
	
	x1 = x1 - buffer
	if(x1 < 0): x1 = 0

	#x2 = x2 + buffer

	cropped_image = img[y1:y2, x1:x2]
	
	print(position)
	print(x1, y1, x2, y2)

	return cropped_image



def store_images(imgs, image_folder, prefix, suffix, zeros = 5):
	
	i = 0 
	for i in range (len(imgs)):
		num = str(i).zfill(zeros)
		filename = image_folder + prefix + num + suffix
		cv.imwrite(filename, imgs[i])


def take_raw_image():
	h = 1080
	w = 1920
	i = 0
	prefix = 'test_image_'
	suffix = '.JPG'

	test_image = cv.imread(test_image_path)


	dictionary = load_dictionary()

	with picamera.PiCamera() as camera:
		camera.start_preview()
		time.sleep(2)
		
		camera.resolution = (w, h)

		while(i < 1):

			with picamera.array.PiRGBArray(camera) as stream:

				camera.capture( stream, format='rgb' )

				output = stream.array

				image = img_pipeline(output)

				images = parse_full_image(test_image, dictionary)

				store_images(images, train_image_folder, prefix, suffix)

				
			i+=1
				
	camera.close()


if __name__ == "__main__":
	take_raw_image()