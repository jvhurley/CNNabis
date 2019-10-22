'''

Author:
	Andrew Dix Hill; https://github.com/AndrewDHill/CNNabis ; andrew.hill@waterboards.ca.gov

Agency:
	California State Water Resource Control Board (SWRCB)

Purpose:
	A tool for downloading NAIP files.

How to use this script:
	how to use the flags

Prerequisites:
	python 3
		tqdm module
Inputs:
	txt or csv file with unique NAIP image identifiers separated by newlines, ex.
	m_4112307_ne_10
	m_4112308_nw_10
	m_4112204_ne_10

	... notice these do not include the timestamp when there were acquired...
Outputs:
	NAIP files to a location you specify.

Next steps:
	Process images into 1024x1024 bit size pieces. See NAIP_img_processing.py

'''

# create a chopped up NAIP image.
import os
import sys
import cv2
from joblib import Parallel, delayed
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_images", required=True,
                help="path of folder to NAIP images to process")
ap.add_argument("-o", "--output_location", required=True,
                help="location to write processed images")
args = vars(ap.parse_args())

# make a folder maker
def batch_folder(tempDir, batch, endName):
	outPath = os.path.join(tempDir, endName, str(batch))
	if not os.path.isdir(outPath):
		os.mkdir(outPath)
	return outPath

# to pass to function
def image_cutter(filePath, inPath, batch):
	FPath, FName = os.path.split(filePath)
	F, ext = os.path.splitext(FName)
	img = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
	(height, width, dim) = img.shape
	#print(f'The depth is {dim}')
	Height_div = height // 1024  #
	Width_div = width // 1024
	if not os.path.exists(os.path.join(inPath, '2016_1024x1024')):
		os.mkdir(os.path.join(inPath, '2016_1024x1024'))
	if not os.path.exists(os.path.join(inPath, '2016_1024x1024_jpg')):
		os.mkdir(os.path.join(inPath, '2016_1024x1024_jpg'))

	for x in range(0,Height_div):
		Xmin, Xmax = x * square, (x + 1) * square
		for y in range(0, Width_div):
			Ymin, Ymax = y * square, (y + 1) * square
			img_chop = img[Xmin:Xmax,Ymin:Ymax,:]
			#print(batch)
			outPath = batch_folder(inPath, batch // 5 , '2016_1024x1024')
			Jpeg_outPath = batch_folder(inPath, batch // 5, '2016_1024x1024_jpg')
			NN = ''.join((outPath, '/', F, '_', str(x), "_", str(y))) + ext
			NN_Jpeg = ''.join((Jpeg_outPath, '/', F, '_', str(x), "_", str(y))) + '.jpg'
			if not os.path.exists(NN) and dim == 4:
				try:
					cv2.imwrite(NN, img_chop)
					pass
				except (KeyboardInterrupt, SystemExit, SystemError):
					if cv2.imread(img_chop).shape != (1024,1024,3):
						os.remove(img_chop)
					sys.exit()
			if dim != 4:
				print(f'The image dimension is {dim}... for {FName}')
			if not os.path.exists(NN_Jpeg):
				cv2.imwrite(NN_Jpeg, img_chop[:,:,:3])
	print(f'Numbers of NAIP images processed: {batch}')

if __name__ == '__main__':

	# Get a list of images to be chopped
	img_dir  = args['input_images']
	files = os.listdir(img_dir)
	FILES=[]
	for f in files:
		FILES.append(os.path.join(img_dir, f))

	outPath = args['output_location']

	# image square dimension
	square = 1024
	batch = 0

	for file in FILES:
		image_cutter(file, outPath, batch)
		batch += 1
