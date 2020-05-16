import numpy as np
import cv2 as cv
import os
import picamera
import picamera.array
import time
import operator


CWD = '/home/pi/Desktop/ChessMate/'
img_folder = CWD + 'data/image_collection/'


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

def img_grayscale(img):
	img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

	return img

#Uses Canny Edge detection and image dialtion to create a 
# edge diagram of the chess board

def edge_detection(img):
	
	kernel = np.ones((5,5), np.uint8)

	CannyBoard = cv.Canny(img, 80, 250)
	DialationBoard = cv.dilate(CannyBoard, kernel, iterations = 3)

	return DialationBoard

def line_transform(img):
	
	hough_lines = cv.HoughLinesP( img, rho = 0.1, theta = np.pi / 90, threshold = 12, minLineLength = 750, maxLineGap = 3)
	return hough_lines 

def draw_lines(img, lines, color = [255, 0, 0], thickness = 5, makeCopy = True):
	if(makeCopy):
		img = np.copy(img)
		img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
		print(img.shape)
	cleaned_list = []
	for line in lines:
		for x1, y1, x2, y2 in line:
			if((abs(y2 - y1) <= 5) and (abs(x2 - x1) in range (500, 800))):
				cleaned_list.append((x1, y1, x2, y2))
				cv.line(img, (x1, y1), (x2, y2), [255,0,0], thickness, cv.LINE_AA)
	print('# of lines detected: ', len(cleaned_list))
	return(img)

def boundaries(line_list):
	temp_line_list = []
	y_avg_list = []

	x_min = 10000
	x_max = 0
	x1_sum = 0
	x2_sum = 0
	y_avg = 0
	count = 0
	y_sum = 0
	y_avg = 0
	i = 0

	for line in line_list:
		for x1, y1, x2, y2 in line:
			if(x1 < x_min):
				x_min = x1

			if(x2 > x_max):
				x_max = x2

			y_avg = (y1 + y2)//2
			temp_line_list.append((x1, x2, y_avg))

			
		

	temp_line_list = sorted(temp_line_list, key = operator.itemgetter(2))


	for line in temp_line_list:
		x1, x2, y = line
		if(count == 0):
			y_avg = y
		if((y <= (y_avg - 50)) or (y >= (y_avg + 50))):
			y_avg_list.append(y_avg)
			y_sum = 0
			count = 0

		else:
			y_sum += y
			count += 1
			y_avg = y_sum//count

				
		x1_sum += x1
		x2_sum += x2
		i += 1			#WHAT IS THIS USED FOR?

	x1_avg = x1_sum//len(line_list)
	x2_avg = x2_sum//len(line_list)


	
	if(count != 0):
		y_avg_list.append(y_avg)

	y_min = min(y_avg_list)
	y_max = max(y_avg_list)
	
	rectangle = (x1_avg, y_min, x2_avg, y_max)
	splits = y_avg_list[1:-1]


	return(rectangle, splits)
	


def rect_parse(rectangle, splits):

	square_count = (len(splits) + 1) * (len(splits) + 1)
	rectangle_list = [[] for i in range(square_count) ]
	x1, y1, x2, y2 = rectangle
	x_step = abs(x2 - x1)// 8
	y_step = abs(y2 - y1)// 8
	v_lines = []
	h_lines = []

	print(rectangle)

	for i in range (0, len(splits) + 2):
		y1_new = y1 + i * y_step + y_step	#h line count is 7+2 = 9
		x1_new = x1 + i * x_step

		h_lines.append( y1_new )
		v_lines.append( x1_new )

	for r in range(0, len(splits) + 1):

		y2_new = r * ( y_step + 1 ) + y1
		y1_new = r * y_step + y1
		
		for c in range (0, len(splits) + 1):

			index = ( r * (len(splits) + 1 ) ) + c
			x2_new = c * (x_step + 1) + x1
			x1_new = c * x_step + x1

			rectangle_list[ index ] = ( x1_new, y1_new, x2_new, y2_new )
	
	for i in range (0, len(splits) + 1):

		if(i == 0):
			for j in range((len(splits) + 1)):
				x1_new = x_step * j + x1
				x2_new = x_step * (j + 1) + x1					

				rectangle_list[j] = (x1_new, y1, x2_new, splits[i])

		elif(i >= (len(splits))):
			for j in range((len(splits) + 1)):
				x1_new = x_step * j + x1
				x2_new = x_step * (j + 1) + x1

				rectangle_list[j] = (x1_new, splits[-1], x2_new, y2)

		else:
			for j in range((len(splits) + 1)):
				x1_new = x_step * j + x1
				x2_new = x_step * (j + 1) + x1

				rectangle_list[j] = (x1_new, splits[i - 1], x2_new, splits[i])

	return (rectangle_list, v_lines, h_lines) 



def draw_final_chessboard( v_lines, h_lines, outer_boundary, img, makeCopy = True):

	if( makeCopy ):
		img = np.copy(img)

	if( len(img.shape) == 2 ):
		img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

	thickness = 10
	x1, y1, x2, y2 = outer_boundary

	# DRAW CROSS SECTION LINES
	color = [255, 0 , 0]
	for i in range( len( v_lines )):
		x = v_lines[ i ]
		cv.line(img, (x, y1), (x, y2), color, thickness, cv.LINE_AA)
	
		y = h_lines[ i ]
		cv.line(img, (x1, y), (x2, y), color, thickness, cv.LINE_AA)

	# DRAW OUTER PERIMETER
	color = [0, 255, 0]
	cv.line(img, (x1, y1), (x2, y1), color, thickness, cv.LINE_AA)
	cv.line(img, (x1, y2), (x2, y2), color, thickness, cv.LINE_AA)
	cv.line(img, (x1, y1), (x1, y2), color, thickness, cv.LINE_AA)
	cv.line(img, (x2, y1), (x2, y2), color, thickness, cv.LINE_AA)

	return img


def get_chessboard_dictionary(rectangle_list):

	letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	
	chessboard_dictionary = {}

	for r in range (0, 8):
		alpha_id = letters[r]
		for c in range(0, 8):
			num_id = str( c + 1 )
			position_string = alpha_id + num_id
			chessboard_dictionary[position_string] = rectangle_list[r * 8 + c]
		
	return chessboard_dictionary

def data_Collection():
	h = 1080
	w = 1920
	i = 0
	prefix = 'img_final_board_'
	suffix = '.JPG' 

	dialation_image = cv.imread('/home/pi/Desktop/ChessMate/data/image_collection/DialationGray.JPG')
	print(dialation_image.shape)
	dialation_image_gray = cv.cvtColor(dialation_image, cv.COLOR_RGB2GRAY)

	with picamera.PiCamera() as camera:
		#camera.start_preview()
		time.sleep(2)
		
		camera.resolution = (w, h)

		while(i < 1):

			with picamera.array.PiRGBArray(camera) as stream:

				camera.capture( stream, format='rgb' )

				output = stream.array

				image = img_pipeline(output)
				filename = img_folder + prefix + str(i) + suffix
				
				image = edge_detection(image)
				
				horizontal_lines = list( line_transform(dialation_image_gray) )
				rect, splits = boundaries( horizontal_lines )

				rectangle_list, v_lines, h_lines = rect_parse(rect, splits)
				chessboard_img = draw_final_chessboard( v_lines, h_lines, rect, dialation_image_gray)

				dictionary = get_chessboard_dictionary( rectangle_list )
				print(dictionary['a8'])


				cv.imwrite(filename, chessboard_img)
				
			i+=1
				
	camera.close()


if __name__ == "__main__":
	data_Collection()
