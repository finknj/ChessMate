#TEST
import numpy as np
import cv2 as cv
import os
import time
import pickle
import warnings

from funtions import *

from threading import Event, Thread, _after_fork

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config = config)


GADFLY_LOOP = False
RUNNING_CAMERA = False


"""
def run_predictions_file_thread(threads, predictions_dictionary, termination_event):
	predictions_thread_event = Event()


	threads.append(Thread(target = update_Chessboard_Dictionary, args = (predictions_dictionary, predictions_thread_event, termination_event, )))
	threads[-1].start()
	
	print('predictions file Thread is Ready')

	return(predictions_thread_event)
"""

def initialize_main_driver_thread(threads, next_move, predictions_dictionary, termination_event):
	engine_thread_event = Event()
	threads.append(Thread(target = update_FEN, args = (next_move, predictions_dictionary, engine_thread_event, termination_event, )))
	threads[-1].start()
	
	print('engine_thread_event is Ready')

	return(engine_thread_event)
	
def initialize_window_display_thread(threads, image,  handle, termination_event):
	window_thread_event = Event()


	threads.append(Thread(target = updateWindow, args = (image, handle, window_thread_event, termination_event, )))
	threads[-1].start()
	

	return(window_thread_event)

def initialize_arm_control_thread(threads, chessboardDictionary, next_move, termination_event):
	arm_control_event = Event()

	threads.append(Thread(target = arm, args = (chessboardDictionary, next_move, arm_control_event, termination_event, )))
	threads[-1].start()

	return(arm_control_event)


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
	cv.destroyAllWindows()
	camera.close()
	session.close()



def main():

	threads = []
	termination_event = Event()
	RUNNING_CAMERA = True
	next_move = ['g1', 'h2']

	guess_array = []
	guess_index = 0
	


	print('Initiailizing Camera')
	camera, dummyFrame = initialize_camera()
	rawCapture = picamera.array.PiRGBArray(camera, size = (1920, 1080) )
	camera.capture(rawCapture, format = 'bgr')


	windowHandle = initializeWindow( rawCapture )

	print('Initializing CNN Model..')
	model = load_trained_model()


	print('Initializing Chessboard Dictionaries')
	chessboardDictionary = create_Chessboard_Dictionary()		#Holds the Current Prediction of all 64 squares
	calibrationDictionary = load_chessboard_dictionary()		#Holds Position Data for all 64 squares


	images = parse_full_image(rawCapture.array, calibrationDictionary)
	predictions = get_new_predictions(model, images, chessboardDictionary)
	

	print('Initializing Worker Threads For Engine Interface')

	window_display_event = initialize_window_display_thread(threads, rawCapture, windowHandle, termination_event)
	chessboard_engine_event =  initialize_main_driver_thread(threads, next_move, chessboardDictionary, termination_event)
	arm_event = initialize_arm_control_thread(threads, chessboardDictionary, next_move, termination_event)
	
	#predictions_file_event = run_predictions_file_thread(threads, chessboardDictionary, termination_event)


	#rawCapture.truncate(0)


	print('\nProgram is ready! Press "r" to execute')

	while((cv.waitKey(30) & 0xFF) != ord('r')): None	#GADFLY Loop (busy/waiting)
	rawCapture.truncate(0)



	print('\nProgram is now executing..')


	while(RUNNING_CAMERA):
		print('space to make move; q to quit')
		GADFLY_LOOP = True

		while(GADFLY_LOOP):		#Gadfly Loop

		

			key_event = (cv.waitKey(10) & 0xFF)
			if(key_event == ord('q')):
				print('esc key command, exiting program')			
				RUNNING_CAMERA = False
				GADFLY_LOOP = False
				

			elif(key_event == ord(' ')):
				print('space key BOOM!')
				rawCapture.truncate(0)
				GADFLY_LOOP = False


		if(RUNNING_CAMERA):
				
			print('Go Go PowerRangers!! (new camera capture)')
			camera.capture(rawCapture, format = 'bgr')
	
			#READ CAMERA FRAME / SEND IMAGE THROUGH PIPELINE / PARSE IMAGE

			frame = rawCapture.array
			frame = img_pipeline(frame)
			imgs = parse_full_image(frame, calibrationDictionary)

			#----------------------------------------
			#Get CNN Predictions / Update Dictionary
			#----------------------------------------

			predictions_array = get_new_predictions(model, imgs, chessboardDictionary)
			guess_array.append(predictions_array)
			guess_index += 1
			time.sleep(1)

			

			if(guess_index >= 2):
				

				average_predictions_array = average_Predictions(guess_array)
				guess_index = 0

				#THREAD EVENTS
			
				print('pre time.sleep')
				chessboard_engine_event.set()

				time.sleep(2)
				print('post time.sleep')

				result = readNextMove()
				next_move = [result[0:2], result[2:4]]
				print('under engine event: ', next_move)
				

				#window_display_event.set()
				#time.sleep(2)
				arm_event.set()
				time.sleep(2)


			rawCapture.truncate(0)

	initiate_shutdown(threads, camera, termination_event)


if __name__ == "__main__":
	main()






