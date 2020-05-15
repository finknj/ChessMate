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
	
	hough_lines = cv.HoughLinesP( img, rho = 0.5, theta = np.pi / 180, threshold = 50, minLineLength = 10, maxLineGap = 10 )
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
	x_min = 10000
	x_max = 0
	x1_sum = 0
	x2_sum = 0
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


	count = 0
	y_sum = 0
	y_avg = 0
	y_avg_list = []

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

		if(i == 0):
			x1_avg = x1_sum//len(line_list)
		if(i == (len(line_list) - 1)):
			x2_avg = x2_sum//len(line_list)
		i += 1
	
	if(count != 0):
		y_avg_list.append(y_avg)

	y_min = min(y_avg_list)
	y_max = max(y_avg_list)
	rectangle = (x1_avg, y_min, x2_avg, y_max)
	splits = y_avg_list[1:-1]

	print(splits)

	return(rectangle, splits)
	


def rect_parse(rectangle, splits, img):
	x1, y1, x2, y2 = rectangle
	square_count = (len(splits) + 1) * (len(splits) + 1)
	rectangle_list = [[] for i in range(square_count) ]

	x_step = abs(x2 - x1)// 8
	x1_new = x1
	x2_new = x2

	print(rectangle)

	for i in range (0, len(splits) + 1):
		if(i == 0):
			for j in range((len(splits) + 1)):
				x1_new = x_step * j + x1
				x2_new = x_step * (j + 1) + x1	

				cv.line(img, (x1_new, y1), (x1_new, y2), [255,0,0], 5, cv.LINE_AA)				

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

	return rectangle_list

#def find_squares():


def data_Collection():
	h = 1080
	w = 1920
	i = 0
	prefix = 'img_hough_lines_'
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
				#image = img_grayscale(image)
				line_list = list(line_transform(dialation_image_gray))



				Line_image = draw_lines(dialation_image_gray, line_list)

				rect, splits = boundaries(line_list)

				rectangle_list = rect_parse(rect, splits, Line_image)


				cv.imwrite(filename, Line_image)
				
			i+=1
				
	camera.close()


if __name__ == "__main__":
	data_Collection()
