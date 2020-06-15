import numpy as np
import cv2 as cv
import os
import picamera
import picamera.array
import time
import pickle
import tensorflow as tf
import warnings

from threading import Event, Thread, _after_fork
from tensorflow.python.util import deprecation



warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

deprecation._PRINT_DEPRECATION_WARNINGS = False


tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config = config)

#==========================================================================================================================

CWD = '/home/pi/Desktop/ChessMate/'
modelPath = CWD + 'data/model/best.h5'
testPath = CWD + 'data/test/'

picklepath = CWD + 'data/pickle/calibration_pickle.p'
picklepath_chessboard = CWD + 'data/pickle/chessboard_pickle.p'

classifications = ["blue_bishop\\", "blue_king\\", "blue_knight\\", "blue_pawn\\", "blue_queen\\", "blue_rook\\", "unoccupied\\",
               "yellow_bishop\\", "yellow_king\\", "yellow_knight\\", "yellow_pawn\\", "yellow_queen\\", "yellow_rook\\"]


IMAGE_SIZE = 128

#==========================================================================================================================
#CAMERA PIPELINE FUNCTIONS
#==========================================================================================================================

def initialize_camera():
	camera = PiCamera()
	camera.resolution = (1920, 1080)
	dummyFrame = PiRGBArray(camera, size = (1920, 1080))
	dummyFrame = img_pipeline(dummyFrame)
	dummyFrame = cv.resize(dummyFrame, (320, 240))
	
	return(camera, dummyFrame)


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

def updateWindow(img, handle, thread_event, termination_event):		#TODO	
	cv.imshow(handle, img)		#CHANGE TO THREAD LATER


#==========================================================================================================================
#Dictionary Functions
#==========================================================================================================================

def load_chessboard_dictionary():

	with open( picklepath, mode = 'rb' ) as f:
		file = pickle.load( f )
		chessboard_dictionary = file['chessboard_pickle']

	return chessboard_dictionary


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
def update_Chessboard_Dictionary(predictions, predictions_dictionary, predictions_event, 
					termination_event, picklepath = picklepath_chessboard):

	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


	for r in range(0, 8):
		alpha_id = letters[r]
		for c in range(0, 8):
			num_id = str(c + 1)
			position_string = alpha_id + num_id
			prediction_value = str( predictions[ r * 8 + c ] )

			predictions_dictionary[ position_string ] = prediction_value

			print( position_string + ': ' + str(predictions_dictionary[ position_string] ) )

	pickle.dump(predictions_dictionary, open( picklepath, 'wb' ) )

def update_Engine(predictions_dictionary, thread_event, termination_event):
	a = 1



#==========================================================================================================================
#CNN Model Functions
#==========================================================================================================================

def load_trained_model():
	return(tf.python.keras.models.load_model('/home/pi/Desktop/ChessMate/data/model/best.h5')





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


	

def get_new_predictions(model, imgs):


	predictions = []

	confidenceLevels = []

	for r in range(0, 8):
		for c in range(0, 8):
			image = imgs[ 63 - (r * 8 + c) ] 
			prediction = model.predict( image[None, :, :, : ])[0]
			predictions.append(np.argmax( prediction ) )
			confidenceLevels.append( prediction )

	return( predictions, confidenceLevels ) 
	
#==========================================================================================================================
#MISC Functions
#==========================================================================================================================

def initializeWindow(dummyFrame):
	windowHandle = 'Camera Pipeline'
	cv.namedWindow(windowHandle, cv.WINDOW_NORMAL)
	cv.resizeWindow(windowHandle, (320, 240))
	cv.imshow(windowHandle)
	return(windowHandle)

































































