import numpy as np
import cv2 as cv
import os
import picamera
import picamera.array
import time
import pickle
import tensorflow as tf
import warnings
import json 
import h5py
from tensorflow.python.util import deprecation
from tensorflow.python.keras.models import load_model

print(tf.VERSION)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

deprecation._PRINT_DEPRECATION_WARNINGS = False


tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config = config)

CWD = '/home/pi/Desktop/ChessMate/'
modelPath = CWD + 'data/model/best.h5'
testPath = CWD + 'data/test/'

picklepath = CWD + 'data/pickle/calibration_pickle.p'
picklepath_chessboard = CWD + 'data/pickle/chessboard_pickle.p'

classifications = ["blue_bishop\\", "blue_king\\", "blue_knight\\", "blue_pawn\\", "blue_queen\\", "blue_rook\\", "unoccupied\\",
               "yellow_bishop\\", "yellow_king\\", "yellow_knight\\", "yellow_pawn\\", "yellow_queen\\", "yellow_rook\\"]


IMAGE_SIZE = 128



def load_CNN_model():
	print('Loading CNN Model...')

	if(os.path.isfile(modelPath)):

		print('Loading..')

		model = load_model( modelPath, compile = False )

		print('CNN Initialiazed skynet pewpew') 
		return model

	else: print('error')

def roi_filter(img):
	rows, cols = img.shape[:2]


	y1 = int( rows * 0.06 )
	y2 = int( rows * 0.81 )

	x1 = int( cols * 0.31 )
	x2 = int( cols * 0.73 )

	img = img[ y1:y2, x1:x2]
	return img

def img_pipeline(img):
	
	img = roi_filter(img)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

	

	return img




def load_chessboard_dictionary():

	with open( picklepath, mode = 'rb' ) as f:
		file = pickle.load( f )
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
			image = cropimage(img, position)
			image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
			image_list[c + r * 8] = image


	return image_list




def cropimage(img, position, buffer = 20):
	
	x1, y1, x2, y2 = position

	y1 = y1 - buffer
	if(y1 < 0): y1 = 0

	y2 = y2 + buffer
	
	x1 = x1 - buffer
	if(x1 < 0): x1 = 0

	x2 = x2 + buffer

	cropped_image = img[y1:y2, x1:x2]

	return cropped_image



def store_images(imgs, image_folder, dictionary, 
		prefix = 'image.', suffix = '.JPG', start_index = 0, zeros = 3):
	
	
	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

	for r in range(0, 8):
		alpha_id = letters[r]
		for c in range(0, 8):
			num_id = str(c + 1)
			position_string = alpha_id + num_id

			filename = image_folder + position_string + suffix


			image = imgs[ 63 - (r * 8 + c) ] 
			cv.imwrite(filename, image)


#Creates an empty dictionary of chessboard locations
def create_Chessboard_Dictionary(picklepath = picklepath_chessboard):
	predictions_dictionary = {}
	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

	for r in range(0, 8):
		alpha_id = letters[r]
		for c in range(0, 8):
			num_id = str(c + 1)
			position_string = alpha_id + num_id
			predictions_dictionary[ position_string ] = 0
			

	pickle.dump(predictions_dictionary, open( picklepath, 'wb' ) )

	return(predictions_dictionary)


#Updates to reflect predictions of each tile
def update_Chessboard_Dictionary(predictions, predictions_dictionary, picklepath = picklepath_chessboard):

	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

	print('\t1\t2\t3\t4\t5\t6\t7\t8\n\n')

	for r in range(0, 8):
		alpha_id = letters[r]
		row_str = alpha_id + '\t'
		
		for c in range(0, 8):
			num_id = str(c + 1)
			position_string = alpha_id + num_id
			prediction_value = str( predictions[ r * 8 + c ] )

			predictions_dictionary[ position_string ] = prediction_value
		
			row_str += prediction_value + '\t'			

			#print( position_string + ': ' + str(predictions_dictionary[ position_string] ) )

		print(row_str)

	pickle.dump(predictions_dictionary, open( picklepath, 'wb' ) )



def update_Predictions(model, imgs):


	predictions = []

	confidenceLevels = []

	for r in range(0, 8):
		for c in range(0, 8):
			image = imgs[ 63 - (r * 8 + c) ] 
			prediction = model.predict( image[None, :, :, : ])[0]
			predictions.append(np.argmax( prediction ) )
			confidenceLevels.append( prediction )

	return( predictions, confidenceLevels ) 
	

def main():

	h = 1080
	w = 1920
	i = 0
	suffix = '.JPG'
	filename = CWD + 'data/image_collection/' + 'cnn_test_image' + suffix

	

	dictionary = load_chessboard_dictionary()
	predictions_Dictionary = create_Chessboard_Dictionary()


	model = load_CNN_model()

	with picamera.PiCamera() as camera:
		camera.start_preview()
		time.sleep(2)

		camera.resolution = (w, h)						
		while( i < 3 ):
			
			with picamera.array.PiRGBArray(camera) as stream:

				camera.capture( stream, format = 'rgb' )

				capture = stream.array

				image = img_pipeline( capture )

				imgs = parse_full_image(image, dictionary)
				store_images(imgs, testPath, dictionary )

				predictions, confidenceLevels = update_Predictions(model, imgs)
				update_Chessboard_Dictionary(predictions, predictions_Dictionary) 
				print(predictions[0])


				#cv.imwrite(filename, image)
			time.sleep(3)
			i += 1
			

	camera.close()


if __name__ == "__main__":
	main()






