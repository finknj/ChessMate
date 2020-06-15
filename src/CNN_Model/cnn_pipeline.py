import numpy as np
import cv2 as cv
import os
import time
import pickle
import warnings

from functions import *
from threading import Event, Thread, _after_fork

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config = config)


def run_predictions_file_thread(threads, predictions, predictions_dictionary, termination_event):
	predictions_thread_event = Event()
	threads.append(Thread(target = update_Chessboard_Dictionary, args = (predictions, predictions_dictionary, predictions_thread_event, termination_event, )))
	threads[-1].start()
	
	print('predictions file Thread is Ready')

	return(predictions_thread_event)


def begin_send_to_engine(threads, predictions_dictionary,termination_event):
	engine_thread_event = Event()
	threads.append(Thread(target = update_Engine, args = (predictions_dictionary, engine_thread_event, termination_event, )))
	threads[-1].start()
	
	print('engine_thread_event is Ready')

	return(engine_thread_event)
	
def run_window_display_thread(threads, image,  handle, termination_event):
	window_thread_event = Event()
	threads.append(Thread(target = updateWindow, args = (image, handle, window_thread_event, termination_event, )))
	threads[-1].start()
	
	print('Window Display Thread is Ready')

	return(window_thread_event)


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
	frame_index = 0

	print('Initiailizing Camera')
	camera, frame = initialize_camera()
	windowHandle = initialize_window( frame )

	print('Initializing CNN Model..')
	model = load_trained_model()

	image = frame.array
	image = img_pipeline(image)
	imgs = parse_full_image(image, calibrationDictionary)

	predictions = get_new_predictions(model, imgs)
	
	print('Initializing Chessboard Dictionaries')
	chessboardDictionary = create_Chessboard_Dictionary()		#Holds the Current Prediction of all 64 squares
	calibrationDictionary = load_Chessboard_Dictionary()		#Holds Position Data for all 64 squares


	print('Initializing Worker Threads For Engine Interface')
	window_display_event = run_window_display_thread(threads, image, windowHandle, termination_event)
	chessboard_engine_event = begin_send_to_engine(threads, chessboardDictionary, termination_event)
	predictions_file_event = run_predictions_file_thread(threads, predictions, prediction_dictionary, termination_event)

	
	
	time.sleep(0.5)	#Warm Up Period
	for frame in camera.capture_continuous(frame, format = 'bgr', muse_video_port = true )

		print('\nProgram is ready! Press 'r' to execute')
		while((cv.waitKey(30) & 0xFF) != ord('r')): None	#GADFLY Loop (busy/waiting)

		print('\nProgram is now executing..')

		while(True):
			
			#Read Camera Frame / Send Image Through Pipeline / Parse Image

			image = frame.array
			image = img_pipeline(image)
			imgs = parse_full_image(image, calibrationDictionary)

			#Get CNN Predictions / Update Dictionary

			predictions = get_new_predictions(model, imgs)
			#update_chessboard_dictionary(predictions, chessboardDictionary)


			#Trigger Thread Events
			
			window_display_event.set()
			predictions_file_event.set()
			chessboard_engine_event.set()
			


			if((cv.waitKey(1) & 0xFF) == 27): break		#ESC Key to Escape Runtime 
			

	initiate_shutdown(threads, camera, termination_event)


if __name__ == "__main__":
	main()






