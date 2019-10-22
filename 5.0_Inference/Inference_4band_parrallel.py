#!/usr/bin/env python


import os
import sys
import numpy as np
import re
from mrcnn.config import Config
import mrcnn.model as modellib
import ast
import fiona
from fiona.crs import from_epsg
import time
import datetime



# todo 
	# make this file run in a main clause
	# add argparser
	
home = os.path.expanduser('~')
media_home = home.replace("/home", "/media")
# Import custom tools
# try:
# 	location = os.path.dirname(os.path.realpath(__file__))
# except NameError:
location = f'{home}/_projects/Cannabis/5.1_gis/gis/HighPriority_wBG_canna_pond_only'
sys.path.append(os.path.join(f'{home}/_projects/Cannabis/5.0_Inference'))  # To find local version
from CNNabis_utils import load_img_container, update_shapefile

# Root directory of the project
ROOT_DIR = os.path.abspath(f"{media_home}/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band")
#ROOT_DIR = os.path.abspath(f"{home}/Documents/_projects/Cannabis/images/2016_NAIP_CNNabis_4band")
#ROOT_DIR = os.path.abspath(f"{media_home}/Storage/images/NAIP_Imagery/2016_NAIP_CNNabis_4band")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(f'{media_home}/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/logs/'
                               'cnnabis_naip_2016_400bg_minified20191003T1710/mask_rcnn_cnnabis_naip_2016_400bg_minified_0047.h5')
#COCO_MODEL_PATH = os.path.join(f'{home}/Documents/_projects/Cannabis/images/2016_NAIP_CNNabis_4band/logs/temp_model2.h5')
#COCO_MODEL_PATH = os.path.join(f'{media_home}/Storage/images/NAIP_Imagery/2016_NAIP_CNNabis_4band/logs/temp_model2.h5')


# ## Configurations
#
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
#
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.


class CNNabisConfig(Config):
	"""Configuration for training on MS COCO.
	Derives from the base Config class and overrides values specific
	to the COCO dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "CNNabis_inference"
	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 1
	# Number of classes (including background)
	NUM_CLASSES = 1 + 7  # CNNabis has 6 classes + 1 back ground
	# Use smaller anchors because our image and objects are small
	RPN_ANCHOR_SCALES = ( 32, 64, 96, 128, 256)
	# (8, 24, 32, 64, 96)  # anchor side in pixels
	#RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
	
	#RPN_TRAIN_ANCHORS_PER_IMAGE = 256
	
	# 4-band modifications.
	# alter utils.py in mrcnn folder, line 364 to leave 4 band in place
	# for 4-band images
	IMAGE_CHANNEL_COUNT = 4
	# adjust the mean pixel value for alpha-band, I just made up 100.0
	# mean pixel caluclated as per the CNNabis_h5py file
	# default values from MaskRCNN were = [123.7, 116.8, 103.9]
	MEAN_PIXEL = np.array([58.87, 66.51, 59.69, 148.97])
	# Detection values
	DETECTION_MIN_CONFIDENCE = 0.3
	# # Detection instances
	# DETECTION_MAX_INSTANCES = 400

	# # Number of ROIs per image to feed to classifier/mask heads
	# # The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# # enough positive proposals to fill this and keep a positive:negative
	# # ratio of 1:3. You can increase the number of proposals by adjusting
	# # the RPN NMS threshold.
	# TRAIN_ROIS_PER_IMAGE = 256

	# # Non-max suppression threshold to filter RPN proposals.
	# # You can increase this during training to generate more proposals.
	# RPN_NMS_THRESHOLD = 0.7

	# # learning rate slow it down after you see the right signs
	# LEARNING_RATE = 0.001

	# # How many anchors per image to use for RPN training
	# RPN_TRAIN_ANCHORS_PER_IMAGE = 256

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = False  # this is a real memory hog, try it at work
	MINI_MASK_SHAPE = (80, 80)  # (56, 56)  # (height, width) of the mini-mask

	# Max number of final detections
	DETECTION_MAX_INSTANCES = 100

	# Ratios of anchors at each cell (width/height)
	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
	RPN_ANCHOR_RATIOS = [0.5, 1, 2]


config = CNNabisConfig()
# for inference change here to increase images per GPU
config.IMAGES_PER_GPU = 10
config.BATCH_SIZE = config.IMAGES_PER_GPU
config.IMAGE_SHAPE = np.array([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, config.IMAGE_CHANNEL_COUNT])
config.display()

# ## Create Model and Load Trained Weights
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)#, exclude=['conv1'])

# ## Class Names
class_names = ['BG', 'outdoor_cannabis', "hoop_greenhouse", "framed_greenhouse", "greenhouse_footprint", "pond",
			   "water_tank", 'bg']

# gens = {'outdoor_cannabis': {}, 'hoop_greenhouse': {}, 'framed_greenhouse': {}, 'greenhouse_footprint': {},
# 		'pond': {}, 'water_tank': {}, 'bg': {}}

# create a list of images or read them in from a file.
# image1 = f'{home}/_projects/Cannabis/Inference/m_3912346_nw_10_h_20160529.tif'
# image2 = f'{home}/_projects/Cannabis/Inference/m_4012456_ne_10_h_20160528.tif'
# image3 = f'{home}/_projects/Cannabis/Inference/m_4012320_se_10_h_20160528.tif'
# image4 = f'{home}/_projects/Cannabis/Inference/m_4012206_se_10_h_20160717.tif'
# image5 = f'{home}/_projects/Cannabis/Inference/m_4012357_sw_10_h_20160528.tif'
# image6 = f'{home}/_projects/Cannabis/Inference/m_3411937_sw_11_h_20160713.tif'
# image7 = f'{home}/_projects/Cannabis/Inference/m_4012339_sw_10_h_20160811.tif'
# img_list = [image2, image1, image3, image4]
# img_list = [image1, image2, image3, image4, image5, image6, image7]

# Directory of images to run detection on
IMAGE_DIR = os.path.join('/media/j5/5TB/NAIP_Imagery/2016')
ll = os.listdir(IMAGE_DIR)
img_list = []
with open(f'{home}/_projects/Cannabis/5.1_gis/gis/selection_areas/High_priority_selection.csv') as trin:
	for row in trin:
		pattern = row.strip()
		for file in ll:
			if re.match(pattern, file):
				#print(file)
				img_list.append(os.path.join(IMAGE_DIR, file))

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

outdoor_cannabis = os.path.join(location, 'outdoor_cannabis.shp')
hoop_greenhouse = os.path.join(location, 'hoop_greenhouse.shp')
framed_greenhouse = os.path.join(location, 'framed_greenhouse.shp')
greenhouse_footprint = os.path.join(location, 'greenhouse_footprint.shp')
pond = os.path.join(location, 'pond.shp')
water_tank = os.path.join(location, 'water_tank.shp')
files = [outdoor_cannabis, hoop_greenhouse, framed_greenhouse, greenhouse_footprint, pond, water_tank]


#print(f'Starting inference')
t0 = time.time()
inference_times = []
progress_counter = 0
###############3
#  assess progress
processed_imgs = []
if not os.path.exists(os.path.join(location, 'progress.csv')):
	with open(os.path.join(location, 'progress.csv'), 'w+') as touch:
		pass
with open(os.path.join(location, 'progress.csv'), 'r+') as prog:
	for row in prog:
		processed_imgs.append(row.strip())
###################3
for image in img_list:
	t1 = time.time()
	iteration = os.path.splitext(os.path.split(image)[1])[0]
	progress_iteration = iteration[:-11]
	print(f"Progress saved in {os.path.join(location, 'progress.csv')}")
	if progress_iteration in processed_imgs:
		print(f"Skipping {iteration}")
		progress_counter += 1
		continue
	np_img, profile, XY_list, container = load_img_container(image)
	print(f'\n\nStarting inference of {iteration}')

	sector_list = []
	sector_count = {}
	sector_batch = config.IMAGES_PER_GPU
	for sector in chunks(XY_list,sector_batch):
		for cnt,sec in enumerate(sector):
			if cnt == 0:
				sector_list = []
			sector_list.append(np_img[sec[0]:sec[0]+1024, sec[1]:sec[1]+1024, :])
			if cnt + 1 == sector_batch:
				results = model.detect(sector_list, verbose=0)
		for sector_cnt in range(len(sector)):
			for count, class_ids in enumerate(results[sector_cnt]['class_ids']):
				sec = sector[sector_cnt]
				score = int(results[sector_cnt]['scores'][count] * 100)
				container[class_ids-1, sec[0]:sec[0]+1024, sec[1]:sec[1]+1024] = \
					np.where(results[sector_cnt]['masks'][:, :, count] == True, \
						np.where(container[class_ids-1, sec[0]:sec[0]+1024, sec[1]:sec[1]+1024] < score, \
							score, \
							container[class_ids-1, sec[0]:sec[0]+1024, sec[1]:sec[1]+1024]), container[class_ids-1, sec[0]:sec[0]+1024, sec[1]:sec[1]+1024])
	print(f'                   Completed inference')
	# tools for writing to shapefile when done.
	print('starting polygon conversions')
	for cnt, target_type in enumerate(class_names):
		# if target_type == 'BG' or target_type == 'bg':
		# "hoop_greenhouse", "framed_greenhouse", "greenhouse_footprint", "pond", "water_tank",
		if target_type == 'BG' or target_type == 'bg' or target_type == 'hoop_greenhouse' or target_type == 'framed_greenhouse' or target_type == 'water_tank' or target_type == 'greenhouse_footprint':
			continue
		shp_poly = update_shapefile(container[cnt - 1, :, :], profile)
		print(f'                 finished {target_type}')

		# attempt write to json and merge later
		#priqnt(shp_poly)
		#input()
		#print('writing results to json files')
		with open(os.path.join(location, target_type + '.json'), 'a') as appender:
			if not shp_poly or shp_poly == [[]]:
				continue
			[appender.write(str(x)) for x in shp_poly[0]]
			#print(f'Completed writing {target_type} json for {iteration}')
	print(f'finished writing results to json files for {iteration}')
	with open(os.path.join(location, 'progress.csv'), 'a') as prog:
		prog.write(progress_iteration + '\n')
	dt = round(time.time() - t1, 0)
	inference_times.append(dt)
	print(f"Elapsed time for {iteration} seconds: {str(datetime.timedelta(seconds=dt))}", )
	progress_counter += 1
	print(f'Completed {progress_counter} of {len(img_list)} ... {round(progress_counter/len(img_list) * 100, 1)}%')
	# todo
	# write progress to file and pickup where we left off


print('Starting write process for all shapefiles')
for cnt, target_type in enumerate(class_names):
	if target_type == 'BG' or target_type == 'bg' or target_type == 'hoop_greenhouse' or target_type == 'framed_greenhouse' or target_type == 'water_tank' or target_type == 'greenhouse_footprint' or target_type == 'outdoor_cannabis':
		continue
	with open(os.path.join(location, target_type + '.json'), 'r') as reader:
		read_json = reader.read().replace("}{", "}, {")
		# if not read_json:
		# 	print('')
		# 	continue
		json_list = list(ast.literal_eval(read_json))
		#d = json.load(json_list)
		with fiona.open(os.path.join(location, target_type + '.shp'), 'w',
						driver='ESRI Shapefile', crs=from_epsg('4326'),
						schema={'properties': [('model_conf', 'int')], 'geometry': 'Polygon'}) as dst:
			dst.writerecords([x for x in json_list])
	print(f'Completed writing {target_type} shapefile')

tfinal = str(datetime.timedelta(seconds=round(time.time() - t0, 0)))
print(f'Inference on {len(img_list)} images took {tfinal} \n   with an average image process time of '
      f'{str(datetime.timedelta(seconds=round(np.mean(inference_times))))}')


