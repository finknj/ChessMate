#TEST
import numpy as np
import cv2 as cv
import os
import picamera
import picamera.array
import time
import pickle
import tensorflow as tf
import warnings
import chess
import chess.engine
import asyncio

from Main_Software.RobotControl import *
from threading import Event, Thread, _after_fork
from tensorflow.python.util import deprecation



warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

deprecation._PRINT_DEPRECATION_WARNINGS = False


tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config = config)

#==========================================================================================================================

CWD = '/home/pi/Desktop/ChessMate/'
modelPath = CWD + 'data/model/best.h5'
testPath = CWD + 'data/test/'
fen_text_path = CWD + 'FEN.txt'

picklepath = CWD + 'data/pickle/calibration_pickle.p'
picklepath_chessboard = CWD + 'data/pickle/chessboard_pickle.p'

classifications = ["blue_bishop\\", "blue_king\\", "blue_knight\\", "blue_pawn\\", "blue_queen\\", "blue_rook\\", "unoccupied\\",
               "yellow_bishop\\", "yellow_king\\", "yellow_knight\\", "yellow_pawn\\", "yellow_queen\\", "yellow_rook\\"]


IMAGE_SIZE = 128



#==========================================================================================================================
#FILE POINTER CLASS
#==================================================================================================================
class FileObject():
	def __init__(self, filepath = fen_text_path):
		self.filepath = filepath
	
	def _open_file(self):
		self.file = open(self.filepath, 'w+')  # 'W+' OVERWRITES OR CREATES NEW FILE

	def _close_file(self):
		self.file.close()

	def writeToFile(self, line_list, open = False, close = False):
		if(open == True): self._open_file()
		self.file.writelines(line_list)
		if(close == True): self._close_file()
	
def readNextMove(filepath = fen_text_path):

	f = open(fen_text_path, 'r')
	result = f.readline()
	f.close()
	return(result)

#==========================================================================================================================
#CAMERA PIPELINE FUNCTIONS
#==========================================================================================================================

def initialize_camera():
	camera = picamera.PiCamera()
	camera.resolution = (1920, 1080)
	dummyFrame = picamera.array.PiRGBArray(camera, size = (1920, 1080) )

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




#==========================================================================================================================
#CNN Model Functions
#==========================================================================================================================

def load_trained_model():
	model = tf.python.keras.models.load_model('/home/pi/Desktop/ChessMate/data/model/best.h5')
	return(model)


def set_predictions_dictionary(predictions, predictions_dictionary):

	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

	print('\t1\t2\t3\t4\t5\t6\t7\t8\n\n')

	for r in range(0, 8):
		alpha_id = letters[r]
		row_str = alpha_id + '\t'
		
		for c in range(0, 8):
			num_id = str(c + 1)
			position_string = alpha_id + num_id

			prediction_value = str(predictions[ r * 8 + c ])

			predictions_dictionary[ position_string ] = prediction_value
		
			row_str += prediction_value + '\t'			

			
		print(row_str)

def get_new_predictions(model, imgs, predictions_dictionary):


	new_predictions = []

	for r in range(0, 8):
		for c in range(0, 8):
			image = imgs[ 63 - (r * 8 + c) ] 
			prediction = model.predict( image[None, :, :, : ])[0]
			new_predictions.append(np.argmax( prediction ) )


	set_predictions_dictionary( new_predictions, predictions_dictionary )

	return( new_predictions ) 


#get_new_predictions
# Turn into numpy array 
# use array.mode to gather best 2/3


#==========================================================================================================================
#FEN NOTATION
#=================================================================================================================


async def getEngineResults(board):
	transport, engine = await chess.engine.popen_uci('stockfish')
	print('here1')
	result = await engine.play(board, chess.engine.Limit(time = 5))
	print('here2 ', result)
	await engine.quit()
	return result.move


#UPDATE PREDICTIONS DICTIONARY // API CALL
def update_FEN(next_move, predictions_dictionary, predictions_event, termination_event, filepath = fen_text_path):
	
	
	board = chess.Board()

	file = FileObject( filepath = filepath )
	
	while( not termination_event.isSet() ):
		predictions_event.wait()
		print('Updating FEN notation')	
		line_list = dictionary_to_text(predictions_dictionary)
		#line_list = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b'
		
		board = chess.Board(line_list)
		valid = board.is_valid()
		print('valid: ' , valid)		
					
		asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
		result = asyncio.run(getEngineResults(board))
		print('exit async')
		#sleep(2)
		result = str(result)
		next_move = [result[0:2], result[2:4]]
		

		file.writeToFile(result, open = True, close = True)

		print('Best Move: ', next_move)

		predictions_event.clear()




def dictionary_to_text(predictions_dictionary):
	
	piece_pairings = ['b', 'k', 'n', 'p', 'q', 'r', 'u', 'B', 'K', 'N','P', 'Q', 'R']
	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	fen_string = ""

	for r in range(8, 0, -1):
		
		num_id = str(r)
		empty_space_loop = False
		empty_space_count = 0
		
		for c in range(0, 8):

			alpha_id = letters[c]
	
			position_string = alpha_id + num_id
			value_key_index = int(predictions_dictionary.get( position_string, '-1' ))
			if(value_key_index != 6):  #IF not an empty space
								
				if(empty_space_loop == True): 
					fen_string += str(empty_space_count)
					empty_space_count = 0
					empty_space_loop = False

				fen_string += str(piece_pairings[ value_key_index ])
				

			else:  #LOOP Through empty spaces
				empty_space_loop = True
				empty_space_count += 1
				if(alpha_id == 'h'): fen_string += str(empty_space_count)
					
				


		if(r != 1): fen_string += '/'

		else: fen_string += ' b'


	print('FEN String: ' , fen_string)
	return(fen_string)

#==========================================================================================================================
#ARM CONTROL FUNCTIONS
#==========================================================================================================================
## ARM STUFF
def arm(chessboardDictionary, next_move, arm_control_event, termination_event):
	RC = RobotControl()
	RC.import_chessboard_dic(chessboardDictionary)
	
	while(not termination_event.isSet()):
		arm_control_event.wait()
		result = readNextMove()
		Move = [result[0:2], result[2:4]]
		print('Arm Stuff: ', Move)
		RC.move_command(Move)	#NOTATION: ["PRESENT STATE", "NEXT STATE"]
		arm_control_event.clear()


#==========================================================================================================================
#MISC Functions
#==========================================================================================================================

def initializeWindow(dummyFrame):
	windowHandle = "CameraPipeline"
	cv.namedWindow(windowHandle, cv.WINDOW_NORMAL)
	cv.resizeWindow(windowHandle, (320, 240))

	image = dummyFrame.array

	image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
	cv.imshow(windowHandle, image)
	return(windowHandle)





def updateWindow(img, handle, thread_event, termination_event):	
	
	while( not termination_event.isSet() ):
		thread_event.wait()
		print('Updating Window')
		
		#cv.destroyAllWindows()
		#cv.namedWindow(handle, cv.WINDOW_NORMAL)
		#cv.resizeWindow(handle, (320, 240))
		
		#image = np.copy(img.array)

		#image = img_pipeline(image)
		#image = cv.resize(image, (320, 240))

		#cv.imwrite( CWD + 'UPDATE_2.JPG', image)



		#img = cv.imread( '/home/pi/Desktop/ChessMate/data/test_images/test_image0.JPG')
		
		#img = cv.resize(img, (320, 240))

		#cv.imshow(handle, img )
						
		#time.sleep(2)
		thread_event.clear()



























































