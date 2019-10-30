#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.



import os
import sys
import random
import math
import numpy as np
import skimage.io
from skimage.measure import find_contours
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import time
import cv2
import glob
from copy import deepcopy
import random
import colorsys
import json
import argparse
from datetime import datetime
from shutil import copyfile



home = os.path.expanduser('~')
media_home = home.replace("/home", "/media")

# Root directory of the project
ROOT_DIR = os.path.abspath(home + "/_projects/Cannabis")

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
#from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/logs/cnnabis_naip_2016_400bg_minified20191003T1710/mask_rcnn_cnnabis_naip_2016_400bg_minified_0024.h5')

# Directory of images to run detection on
IMAGE_DIR = '/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/train2016'

class CNNabisConfig(Config):
	"""Configuration for inference on MS COCO.
	Derives from the base Config class and overrides values specific
	to the COCO dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "CNNabis_inference"
	# Uncomment to train on 8 GPUs (default is 1)
	# GPU_COUNT = 8
	IMAGES_PER_GPU = 2
	# Number of classes (including background)
	NUM_CLASSES = 1 + 7  # CNNabis has 6 classes + 1 back ground
	# NUM_CLASSES = 1 + 6  # CNNabis has 6 classes 
	# Use smaller anchors because our image and objects are small
	#RPN_ANCHOR_SCALES = ( 8, 16, 32, 64, 96, 128)  # failed?!
	#RPN_ANCHOR_SCALES = ( 16, 64, 128, 256, 312)  # anchor side in pixels
	#RPN_ANCHOR_SCALES = ( 32, 64, 128, 256, 384)  # anchor side in pixels
	#RPN_ANCHOR_SCALES = ( 32, 48, 64, 128, 256)  # not small enough to big enough
	RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # The RPN ratios allow us to capture smaller objects, this setting seems to be a good fit for the imagery, (ponds, and outdoor grows at least)
	#RPN_ANCHOR_SCALES = ( 64, 256, 384, 512, 768)  # this is too big!
	# Detection values
	DETECTION_MIN_CONFIDENCE = 0.01
	# switch to ResNet 50 instead of 101
	#BACKBONE = "resnet50"
	# 4-band modifications.
	# alter utils.py in mrcnn folder, line 364 to leave 4 band in place
	# for 4-band images
	IMAGE_CHANNEL_COUNT = 4
	# adjust the mean pixel value for alpha-band
	# mean pixel caluclated as per the CNNabis_h5py file
	# default values from MaskRCNN were = [123.7, 116.8, 103.9]
	MEAN_PIXEL = np.array([58.87, 66.51, 59.69, 148.97])
	# Detection instances
	DETECTION_MAX_INSTANCES = 80
	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = False  # this is a real memory hog, try it at work
	MINI_MASK_SHAPE = (80, 80)  # (56, 56)  # (height, width) of the mini-mask
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
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# We'll have to figure out the right order... it may come from train or val... currently these don't match :(
class_names = ['BG', 'outdoor_cannabis', "hoop_greenhouse", "framed_greenhouse", "greenhouse_footprint", "pond", "water_tank", "bg"]
# class_names = ['BG', 'outdoor_cannabis', "hoop_greenhouse", "framed_greenhouse", "greenhouse_footprint", "pond", "water_tank"] # use this for the temp model

# Load a random image from the images folder
file_names = glob.glob(os.path.join(IMAGE_DIR, '*.tif'))

# here we are going to keep track of the images that we've already looked at.
# we need to link this up to an actual file later on. TODO
reviewed_images = []

def capture_choice(result):
	acceptable = ['y', 'n', 'r']
	user_input = input("Should this image be included in annotations? 'y/n' or with review 'r'")
	if user_input not in acceptable:
		print("Please choose y, n, or r")
		user_input = capture_choice(result)
	if user_input == 'y':
		return 'pass', result
	if user_input == 'n':
		return 'rejected', None
	if user_input == 'r':
		return 'send for review', result


def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1,
								  image[:, :, c] *
								  (1 - alpha) + alpha * color[c] * 255,
								  image[:, :, c])
	return image

def display_instances(image, boxes, masks, class_ids, class_names,
					  scores=None, title="",
					  ax=None,
					  show_mask=True, show_bbox=True,
					  colors=None, captions=None,
					  choices=None, rez=None, fig=None):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
	title: (optional) Figure title
	show_mask, show_bbox: To show masks and bounding boxes or not
	figsize: (optional) the size of the image
	colors: (optional) An array or colors to use with each object
	captions: (optional) A list of strings to use as captions for each object
	"""

	# Number of instances
	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

	# Generate random colors
	colors = colors or random_colors(N)

	ax.clear()
	# Show area outside image boundaries.
	height, width = image.shape[:2]
	ax.set_ylim(height + 10, -10)
	ax.set_xlim(-10, width + 10)
	ax.axis('off')
	ax.set_title(title)

	masked_image = image.astype(np.uint32).copy()

	for i in range(N):
		# if class_ids[i] != 1 or class_ids[i] != 5:
		# 	# skip displaying these categories
		# 	continue
		# make categories_of_interest greater than 0 to signify we've at least one good box
		# categories_of_interest += 1

		color = colors[i]

		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		y1, x1, y2, x2 = boxes[i]
		if show_bbox:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
								alpha=0.7, linestyle="dashed",
								edgecolor=color, facecolor='none')
			ax.add_patch(p)

		# Label
		if not captions:
			class_id = class_ids[i]
			score = scores[i] if scores is not None else None
			label = class_names[class_id]
			x = random.randint(x1, (x1 + x2) // 2)
			caption = "{} {:.3f}".format(label, score) if score else label
		else:
			caption = captions[i]
		ax.text(x1, y1 + 8, caption,
				color='w', size=11, backgroundcolor="none")

		# Mask
		mask = masks[:, :, i]
		if show_mask:
			masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=color)
			ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8))
	# fig.canvas.draw()
	plt.draw()
	choices.append(capture_choice(rez))

def Inference_now(file_names, IMAGE_DIR, model):
	imgs_for_inference = []
	titles = []
	choices = []
	for img in range(config.IMAGES_PER_GPU):
		img_choice = random.choice(file_names)
		imgs_for_inference.append(skimage.io.imread(os.path.join(IMAGE_DIR, img_choice )))
		#imgs_for_inference.append(cv2.imread(os.path.join(IMAGE_DIR, img_choice), cv2.IMREAD_UNCHANGED))
		titles.append(img_choice)
		print(img_choice)
	results = model.detect(imgs_for_inference, verbose=1)
	fig, ax = plt.subplots(1, figsize=(10, 10))
	plt.show(block=False)
	# Visualize resultsq
	for count, rez in enumerate(results):
		if not rez['rois'].shape[0]:
			choices.append(('rejected', None))
			continue
		##############  Pickel rez for demo or experimentation purposes
		#import pickle
		#direct = os.getcwd()
		#with open(os.path.join(direct, str(count) + '.pickle'), 'wb') as handle:
		#	pickle.dump(rez, handle, protocol=pickle.HIGHEST_PROTOCOL)
		##############  Pickel rez for demo or experimentation purposes
		# reorder bands
		color_img = imgs_for_inference[count][:, :, :3]
		# If no axis is passed, create one and automatically call show()

		display_instances(color_img, rez['rois'], rez['masks'], rez['class_ids'],
				class_names, rez['scores'], show_mask=False, title=titles[count], choices=choices, rez=rez,
						  fig=fig, ax=ax)
	return dict(zip(titles, choices))

def return_passed_dict(Cannavision_result, image_full_name, images, verified_data, img_count):
	instance = Cannavision_result
	path, file_name = os.path.split(image_full_name)
	height, width = instance['masks'].shape[:2]
	files = []
	Cannavision_result['verts'] = [None] * len(Cannavision_result['class_ids'])
	Ann_dict = []
	for cnt, cl_id in enumerate(Cannavision_result['class_ids']):
		if cl_id == 1 or cl_id == 5:
			# here we handle the image info and add to the verified_data object for writing later.
			if file_name in images:
				image_id = images[file_name]['id']
			else:
				img_dict = {"file_name": file_name, "height": height, "width": width, "id": img_count}
				images[file_name] = img_dict
				verified_data['images'].append(img_dict)
				image_id = img_count
				img_count += 1
			# now we build the annotation to append to the annotations object
			bbox = Cannavision_result['rois'][cnt].tolist()
			category_id = Cannavision_result['class_ids'][cnt]
			# if category id is 5 we need to change it to 2
			if category_id == 5:
				category_id = 2

			# Mask Polygon
			# Pad to ensure proper polygons for masks that touch image edges.
			mask = Cannavision_result['masks'][:, :, cnt]
			###   enter fix here.
			padded_mask = np.zeros(
				(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
			padded_mask[1:-1, 1:-1] = mask
			contours = find_contours(padded_mask, 0.5)
			for verts in contours:
				# Subtract the padding and flip (y, x) to (x, y)
				verts = np.fliplr(verts) - 1
				segmentation_raw = verts.tolist()
				even = [even[0] for even in segmentation_raw]
				odd = [odd[1] for odd in segmentation_raw]
				segmentation = [None] * (len(even) + len(odd))
				segmentation[::2] = [x for x in even]
				segmentation[1::2] = [x for x in odd]
			Cannavision_result['verts'][cnt] = segmentation
			Ann_dict.append({"segmentation": [Cannavision_result['verts'][cnt]], "image_id": image_id,
			            "bbox": bbox, "category_id": category_id})
			# here are the files that may need review
			files.append(file_name)
		else:
			continue
	return Ann_dict, img_count, files


if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-o", "--outfile", default=home + '/_projects/Cannabis/3.4_Annotation_Augmentation/temp_passed.json',
	                help="Path of folder containing all annotations files to merge")
	ap.add_argument("-r", "--review_outfile", default=home + '/_projects/Cannabis/3.4_Annotation_Augmentation/temp_review.json',
	                help="Path of folder containing all annotations files to merge")
	args = vars(ap.parse_args())

	# read in annotation files
	outfile = args['outfile']
	review_outfile = args['review_outfile']
	image_review_folder = os.path.split(review_outfile)[0] + "/images_for_review"
	# create image review folder if doesn't exist
	if not os.path.exists(image_review_folder):
		os.mkdir(image_review_folder)
	progress = os.path.split(review_outfile)[0] + "/processed_images.txt"

	# if a review annotation file exists, read it in and save it for merging with results later
	# we need to advance the image counter if this file exists. THe pass and review files should have the same images
	# so you shouldn't have to do this twice
	if os.path.exists(outfile):
		with open(outfile, 'r') as T:
			TEMP = T.read()
		verified_data = json.loads(TEMP)
		annotations = deepcopy(verified_data['annotations'])
	else:
		verified_data = {'images': [], 'type': 'instances', 'annotations': [],
		 'categories': [{'supercategory': 'none', 'id': 1, 'name': 'outdoor_cannabis'},
		                {'supercategory': 'none', 'id': 2, 'name': 'pond'}]}

	# if a needs review annotation file exists, read it in
	if os.path.exists(review_outfile):
		with open(review_outfile, 'r') as T:
			TEMP = T.read()
		verified_data_review = json.loads(TEMP)
		annotations_for_review = deepcopy(verified_data_review['annotations'])
	else:
		annotations_for_review = []

	# keep track of the images we've already looked at using a txt file
	if os.path.exists(progress):
		with open(progress, 'r') as progresser:
			progress_list = [line.strip() for line in progresser]
	else:
		progress_list = []
		with open(progress, 'w') as temp_prog:
			pass

	# use the images object to use a number and retrieve a name
	images = {}
	img_count = 1
	for im in verified_data['images']:
		images[im['file_name']] = im
		img_count += 1

	# use the cats object to use a number and retrieve a name
	cats = {}
	for cat in verified_data['categories']:
		cats[cat['id']] = cat['name']

	state = ''
	while state != 'n':
		choices = Inference_now(file_names, IMAGE_DIR, model)
		#print(choices)
		for count, key in enumerate(choices):
			if choices[key][0] == 'rejected':
				file = os.path.split(key)[1]
				progress_list.append(file)
				continue
			elif choices[key][0] == 'send for review':
				ann_dict, img_count, file = return_passed_dict(choices[key][1], key, images, verified_data, img_count)
				annotations_for_review.extend(ann_dict)
				file_n = os.path.join(IMAGE_DIR, file[0])
				copyfile(file_n, os.path.join(image_review_folder, file[0]))
				if file not in progress_list:
					progress_list.append(file[0])
			elif choices[key][0] == 'pass':
				ann_dict, img_count, file = return_passed_dict(choices[key][1], key, images, verified_data, img_count)
				#print(ann_dict)
				annotations.extend(ann_dict)
				if file not in progress_list:
					progress_list.append(file[0])
		state = input("Continue? (y/n)")
		plt.close()

	# Create the structure to write out pass annotations file
	data = {}
	data['type'] = "instances"
	data['images'] = verified_data['images']
	data['annotations'] = annotations
	data['categories'] = verified_data['categories']
	# if outfile exists, read it in and merge with data
	try:
		with open(outfile, 'w') as W:
			W.write(str(data).replace('\'', '\"').replace(' ', ''))
	except:
		print("Couldn't write to file")
	# Create the structure to write out pass annotations file
	data_for_review = {}
	data_for_review['type'] = "instances"
	data_for_review['images'] = verified_data['images']
	data_for_review['annotations'] = annotations_for_review
	data_for_review['categories'] = verified_data['categories']
	try:
		with open(review_outfile, 'w') as W:
			W.write(str(data_for_review).replace('\'', '\"').replace(' ', ''))
	except:
		print("Couldn't write to file")

	# read in the old progress so we can update only the new files to the document. THis is probably overkill...
	with open(progress, 'r') as progresser:
		progress_prev = [line.strip() for line in progresser]
	with open(progress, 'a') as progress_appender:
		for file in progress_list:
			if file not in progress_prev:
				progress_appender.write(file + '\n')



