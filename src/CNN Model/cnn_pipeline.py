import numpy as np
import cv2 as cv
import os
import picamera
import picamera.array
import time
import pickle
import tensorflow as tf
import warnings

from functions import *
from threading import Event, Thread, _after_fork
from tensorflow.python.util import deprecation
from tensorflow.python.keras.models import load_model


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

deprecation._PRINT_DEPRECATION_WARNINGS = False


tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config = config)


def run_predictions_file_thread(threads, predictions_dictionary,termination_event):


def begin_send_to_engine(threads, predictions_dictionary,termination_event):
	


def join_threads(threads, termination_event):
	print('Joining threads..')

	termination_event.set()
	for thread in threads:
		thread.join(timeout = 0.01)
		thread._delete()
	
	_after_fork()
	print('Threads are joined.')

def initiate_shutdown(threads, camera, termination_event):
	print('Initiating Shutdown..')
	
	join_threads(threads, termination_event)
	terminate_camera(camera)
	session.close()

def main():

	threads = []
	initial_time = 0
	termination_event = Event()

	print('Initiailizing Camera')
	camera, dummyFrame = initialize_camera()


	print('Initializing CNN Model..')
	model = load_trained_model()

	
	print('Initializing Chessboard Dictionaries')
	chessboardDictionary = create_Chessboard_Dictionary()


	print('Initializing Worker Threads For Engine Interface')
	chessboard_engine_event = begin_send_to_engine(threads, chessboardDictionary, termination_event)
	predictions_file_event = run_predictions_file_thread(threads, chessboardDictionary, termination_event)


	print('Initializing window threads')
	initialize_window( dummyFrame )
	

	if( ):
		print('\nProgram is ready! Press 'r' to execute')
		while((cv.waitKey(30) & 0xFF) != ord('r')): None	#GADFLY Loop (busy/waiting)

		print('\nProgram is now executing..')
		while(True):
			
			#Read Camera Frame / Send Image Through Pipeline / Parse Image


			#Get CNN Predictions / Update Dictionary


			#Stockfish11 API Call



			if((cv.waitKey(1) & 0xFF) == 27): break		#ESC Key to Escape Runtime 
			


	else: print('unable to open camera')

	initiate_shutdown(threads, camera, termination_event)


if __name__ == "__main__":
	main()






