import numpy as np
import cv2 as cv
import os
import time
import pickle
import warnings

from funtions import *
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
	#image = np.empty( shape = (1920, 1080, 3) )
	
	camera, dummyFrame = initialize_camera()

	rawCapture = picamera.array.PiRGBArray(camera, size = (1920, 1080) )
	camera.capture(rawCapture, format = 'bgr')


	windowHandle = initializeWindow( rawCapture )

	print('Initializing CNN Model..')
	#model = load_trained_model()



	print('Initializing Chessboard Dictionaries')
	chessboardDictionary = create_Chessboard_Dictionary()		#Holds the Current Prediction of all 64 squares
	calibrationDictionary = load_chessboard_dictionary()		#Holds Position Data for all 64 squares


	predictions = np.zeros( shape = (64) )	

	print('Initializing Worker Threads For Engine Interface')
	window_display_event = run_window_display_thread(threads, rawCapture, windowHandle, termination_event)
	#chessboard_engine_event = begin_send_to_engine(threads, chessboardDictionary, termination_event)
	#predictions_file_event = run_predictions_file_thread(threads, predictions, chessboardDictionary, termination_event)


	rawCapture.truncate(0)
	#raw_Capture = picamera.array.PiRGBArray(camera, size = (1920, 1080) )
	
	#time.sleep(1)	#Warm Up Period

	print('\nProgram is ready! Press "r" to execute')

	while((cv.waitKey(30) & 0xFF) != ord('r')):	#GADFLY Loop (busy/waiting)
		rawCapture.truncate(0)

	print('\nProgram is now executing..')
	while(True):
		print('ready to capture')
		

		#if((cv.waitKey(10) & 0xFF) == 27): 
		#	print('esc key command, exiting program')			
		#	break		#ESC Key to Escape Runtime 

		cv.waitKey(0)
		#while((cv.waitKey(10) & 0xFF) != ord('q')):
		#	rawCapture.truncate(0)
		print('Go Go PowerRangers!! (new camera capture)')
		camera.capture(rawCapture, format = 'bgr')
		
		window_display_event.set()
		time.sleep(1)

		#READ CAMERA FRAME / SEND IMAGE THROUGH PIPELINE / PARSE IMAGE

		frame = rawCapture.array


		frame = img_pipeline(frame)
		#imgs = parse_full_image(frame, calibrationDictionary)

		#Get CNN Predictions / Update Dictionary

		#predictions = get_new_predictions(model, imgs)
		#update_chessboard_dictionary(predictions, chessboardDictionary)


		#Trigger Thread Events
		
		
		#predictions_file_event.set()
		#chessboard_engine_event.set()
		

		
		rawCapture.truncate(0)

	initiate_shutdown(threads, camera, termination_event)


if __name__ == "__main__":
	main()






