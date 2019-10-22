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

# base python stuff
import time
import urllib.request
import re
import os
import sys
import argparse
from tqdm import tqdm


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir_folder", required=True,
                help="Path of folder to save images")
ap.add_argument("-l", "--list_of_images", required=True,
                help="file with list of images to download")
args = vars(ap.parse_args())


location = args['dir_folder']
if not os.path.exists(location):
	os.mkdir(location)

# Base url for CNRA NAIP 2016 tifs
url = 'http://gisarchive.cnra.ca.gov/iso/ImageryBaseMapsLandCover/NAIP/naip2016/NAIP_2016_DOQQ/Data/TIFF/'

currentFiles = os.listdir(location)
TheNamedList = args['list_of_images']
THE_LIST = {}
CNRA_pages = {}

if __name__ == '__main__':
	with open(os.path.join(location, TheNamedList), 'r') as reader:
		count = 0
		for line in reader.readlines():
			image_ID = line.strip()
			CNRA_page_location = image_ID[2:7]
			CNRA_page = os.path.join(url, CNRA_page_location)
			if CNRA_page not in THE_LIST:
				THE_LIST[CNRA_page] = [image_ID]
			else:
				THE_LIST[CNRA_page].append(image_ID)
			count += 1
	
	with tqdm(total=count) as pbar:
		for key,values in THE_LIST.items():
			#print(key, values)
			temp = urllib.request.urlopen(key)
			result = str(temp.read())
			for value in values:
				pattern = re.search(value + '(\w*.tif)', result)
				if pattern:
					pMatch = pattern.group(0)
					img_url = os.path.join(key,pMatch).replace('\\','/')
					img_save = os.path.join(location,pMatch)
					if not os.path.exists(img_save):
						try:
							urllib.request.urlretrieve(img_url, img_save)
							#print(f'saved {img_save}')
						except (KeyboardInterrupt, SystemExit, SystemError):
							os.remove(img_save)
							sys.exit()
					#else:
						#print(f'The image : {img_save} \nalready exists\n')
				pbar.update(1)
