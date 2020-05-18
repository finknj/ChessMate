





#=============================================================================== 
# IMPORT_LIBRARIES 
#=============================================================================== 
import os 
# dealing with directories 
import cv2 as cv 
# dealing with images 
import numpy as np 
# dealing with arrays 
#=============================================================================== 
# VARIABLE_DECLARATIONS 
#=============================================================================== 
CWD = '/home/pi/Desktop/ChessMate/'
#=============================================================================== 
# FUNCTION_DEFINITIONS 
#=============================================================================== 

#Function to rename image files for easier parsing 


def rename_img_files( img_dir, img_base_name ):
	list_dir_seg = np.array( os.listdir( img_dir ) )
	i = 0 
	for filename in list_dir_seg:
		index_str = str( i ).zfill( 4 ) 
		dst = img_base_name + '.' + index_str + ".JPG" 
		src = img_dir + "/" + filename 
		dst = img_dir + "/" + dst
		os.rename( src, dst ) 
		i += 1 
#=============================================================================== 
# RENAMING_FILES 
#=============================================================================== 
img_folder_path = CWD + '/data/train/blue_bishop' 
rename_img_files( img_folder_path, 'blue_bishop' )

img_folder_path = CWD + '/data/train/blue_king' 
rename_img_files( img_folder_path, 'blue_king' )

img_folder_path = CWD + '/data/train/blue_knight' 
rename_img_files( img_folder_path, 'blue_knight' )

img_folder_path = CWD + '/data/train/blue_pawn' 
rename_img_files( img_folder_path, 'blue_pawn' )

img_folder_path = CWD + '/data/train/blue_queen' 
rename_img_files( img_folder_path, 'blue_queen' )

img_folder_path = CWD + '/data/train/blue_rook' 
rename_img_files( img_folder_path, 'blue_rook' )
 
img_folder_path = CWD + '/data/train/unoccupied' 
rename_img_files( img_folder_path, 'unoccupied' )

img_folder_path = CWD + '/data/train/yellow_bishop' 
rename_img_files( img_folder_path, 'yellow_bishop' )

img_folder_path = CWD + '/data/train/yellow_king' 
rename_img_files( img_folder_path, 'yellow_king' )

img_folder_path = CWD + '/data/train/yellow_knight' 
rename_img_files( img_folder_path, 'yellow_knight' )

img_folder_path = CWD + '/data/train/yellow_pawn' 
rename_img_files( img_folder_path, 'yellow_pawn' )

img_folder_path = CWD + '/data/train/yellow_queen' 
rename_img_files( img_folder_path, 'yellow_queen' ) 

img_folder_path = CWD + '/data/train/yellow_rook' 
rename_img_files( img_folder_path, 'yellow_rook' ) 
#===============================================================================

