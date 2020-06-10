import os
import pickle
import cv2 as cv
import numpy as np
import picamera
import picamera.array
import time
import glob
from tqdm import tqdm

CWD = '/home/pi/Desktop/'
train_image_folder = CWD + 'train/'
temp_image_folder = CWD + 'temp/'
picklepath = CWD + 'ChessMate/' + 'data/pickle/chessboard_pickle.p'

IMAGE_SIZE = 128



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

	y2 = y2 + (buffer - 5)
	
	x1 = x1 - buffer
	if(x1 < 0): x1 = 0

	x2 = x2 + (buffer - 5)

	cropped_image = img[y1:y2, x1:x2]

	return cropped_image



def store_images(imgs, image_folder, prefix, suffix, start_index, zeros = 6):
	
	i = start_index 
	for i in range (start_index, len(imgs) + start_index):
		num = str(i).zfill(zeros)
		filename = image_folder + prefix + num + suffix
		image = imgs[i - start_index]
		image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
		print(i - start_index)
		print(filename)
		cv.imwrite(filename, image)


def take_raw_image():
	h = 1080
	w = 1920
	i = 0
	prefix = 'test_image_'
	suffix = '.JPG'

	dictionary = load_dictionary()

	if( os.path.isdir(temp_image_folder) == True ):
		
		image_file_path = glob.glob(temp_image_folder + '*')
		print(len(image_file_path))
		for file_path in tqdm(image_file_path):


			image = cv.imread(file_path)

			images = parse_full_image(image, dictionary)

			store_images(images, train_image_folder, prefix, suffix, i * 64)


			i+=1
				


if __name__ == "__main__":
	take_raw_image()