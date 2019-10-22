"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python3 coco.py train --dataset=/path/to/coco/ --model=coco

	# Train a new model starting from ImageNet weights. Also auto download COCO dataset
	python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

	# Continue training a model that you had trained earlier
	python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

	# Continue training the last model you trained
	python3 coco.py train --dataset=/path/to/coco/ --model=last

	# Run COCO evaluatoin on the last model you trained
	python3 coco.py evaluate --dataset=/path/to/coco/ --model=last

	# In the instance of CNNabis training be sure to update to the latest model
	#  python ./samples/CNNAbis_test_train.py train 
	 	 	  --dataset=/media/robot/Storage/images/NAIP_Imagery/2016_NAIP_CNNabis_train
	 	 	  --model=last
	# to begin with:
		 python CNNAbis_test_train_4band.py train --dataset=/media/robot/Storage/images/NAIP_Imagery/2016_NAIP_CNNabis_4band --model=../mask_rcnn_coco.h5
	# followed by:
	python CNNAbis_test_train_4band.py train --dataset=/media/robot/Storage/images/NAIP_Imagery/2016_NAIP_CNNabis_4band --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

home = os.path.expanduser('~')
media_home = home.replace("/home", "/media")

# Root directory of the project
ROOT_DIR = os.path.abspath(f"{home}/_projects/Cannabis/")

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#DEFAULT_LOGS_DIR = os.path.join(f'{home}/_projects/Cannabis/images/2016_NAIP_CNNabis_4band', "logs")
DEFAULT_LOGS_DIR = os.path.join(f'{media_home}/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band', "logs")
DEFAULT_DATASET_YEAR = "2016"

############################################################
#  Configurations
############################################################


class CNNabisConfig(Config):
	"""Configuration for training on MS COCO.
	Derives from the base Config class and overrides values specific
	to the COCO dataset.
	"""
	# Give the configuration a recognizable name
	# NAME = "CNNabis_NAIP_2016_400BG_minified"
	NAME = "CannaVision_400BGmini_80TRPI"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	# IMAGES_PER_GPU = 1
	IMAGES_PER_GPU = 2

	# Uncomment to train on 8 GPUs (default is 1)
	# GPU_COUNT = 8

	# Number of classes (including background)
	NUM_CLASSES = 1 + 7  # CNNabis has 6 classes + 1 back ground

	# Number of training steps per epoch
	# This doesn't need to match the size of the training set. Tensorboard
	# updates are saved at the end of each epoch, so setting this to a
	# smaller number means getting more frequent TensorBoard updates.
	# Validation stats are also calculated at each epoch end and they
	# might take a while, so don't set this too small to avoid spending
	# a lot of time on validation stats.
	# if it does match the training dataset size, you will know if you have all
	# the images on the first epoch.
	STEPS_PER_EPOCH = 6586

	# Number of validation steps to run at the end of every training epoch.
	# A bigger number improves accuracy of validation stats, but slows
	# down the training.
	VALIDATION_STEPS = 100

	# Use smaller anchors because our image and objects are small
	RPN_ANCHOR_SCALES = (32, 64, 96, 128, 256)  # anchor side in pixels
	#RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

	# 4-band modifications.
	# alter utils.py in mrcnn folder, line 364 to leave 4 band in place
	# for 4-band images
	IMAGE_CHANNEL_COUNT = 4
	# adjust the mean pixel value for alpha-band, I just made up 100.0
	# mean pixel caluclated as per the CNNabis_h5py file
	# default values from MaskRCNN were = [123.7, 116.8, 103.9]
	MEAN_PIXEL = np.array([58.87, 66.51, 59.69, 148.97])

	# Detection values
	# DETECTION_MIN_CONFIDENCE = 0.7

	# switch to ResNet 50 instead of 101
	#BACKBONE = "resnet50"
	
	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 80

	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more proposals.
	RPN_NMS_THRESHOLD = 0.7

	# learning rate slow it down after you see the right signs
	LEARNING_RATE = 0.001

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 320
	
	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = False  # this is a real memory hog, try it at work
	MINI_MASK_SHAPE = (80, 80)  # (56, 56)  # (height, width) of the mini-mask
	
	# Max number of final detections
	DETECTION_MAX_INSTANCES = 300
	
	# Ratios of anchors at each cell (width/height)
	# A value of 1 represents a square anchor, and 0.5 is a wide anchor
	RPN_ANCHOR_RATIOS = [0.5, 1, 2]


############################################################
#  Dataset
############################################################

class CNNabisDataset(utils.Dataset):
	def load_coco(self, dataset_dir, subset, year=2016, class_ids=None,
	              class_map=None, return_coco=False):
		"""Load a subset of the COCO dataset.
		dataset_dir: The root directory of the COCO dataset.
		subset: What to load (train, val, minival, valminusminival)
		year: What dataset year to load (2014, 2017) as a string, not an integer
		class_ids: If provided, only loads images that have the given classes.
		class_map: TODO: Not implemented yet. Supports maping classes from
			different datasets to the same class ID.
		return_coco: If True, returns the COCO object.
		"""

		coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
		image_dir = "{}/{}{}".format(dataset_dir, subset, year)

		# Load all classes or a subset?
		if not class_ids:
			# All classes
			class_ids = sorted(coco.getCatIds())

		# All images or a subset?
		if class_ids:
			image_ids = []
			for id in class_ids:
				image_ids.extend(list(coco.getImgIds(catIds=[id])))
			# Remove duplicates
			image_ids = list(set(image_ids))
		else:
			# All images
			image_ids = list(coco.imgs.keys())

		# Add classes
		for i in class_ids:
			self.add_class("CNNabis", i, coco.loadCats(i)[0]["name"])

		# Add images
		for i in image_ids:
			self.add_image(
				source="CNNabis", image_id=i,
				path=os.path.join(image_dir, coco.imgs[i]['file_name']),
				width=coco.imgs[i]["width"],
				height=coco.imgs[i]["height"],
				annotations=coco.imgToAnns[i])
		# print(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))

		if return_coco:
			return coco

	def load_mask(self, image_id):
		"""Load instance masks for the given image.
		Different datasets use different ways to store masks. This
		function converts the different mask format to one format
		in the form of a bitmap [height, width, instances].
		Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a COCO image, delegate to parent class.
		image_info = {}
		image_info = self.image_info[image_id]

		instance_masks = []
		class_ids = []
		annotations = self.image_info[image_id]["annotations"]
		# Build mask of shape [height, width, instance_count] and list
		# of class IDs that correspond to each channel of the mask.
		for annotation in annotations:
			class_id = annotation['category_id']

			# self.map_source_class_id(
			# "CNNabis.{}".format(annotation['category_id']))
			if class_id:
				m = self.annToMask(annotation, image_info["height"],image_info["width"])
				# Some objects are so small that they're less than 1 pixel area
				# and end up rounded out. Skip those objects.
				if m.max() < 1:
					continue
				# Is it a crowd? If so, use a negative class ID.
				if annotation['iscrowd']:
					# Use negative class ID for crowds
					class_id *= -1
					# For crowd masks, annToMask() sometimes returns a mask
					# smaller than the given dimensions. If so, resize it.
					if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
						m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
				instance_masks.append(m)
				class_ids.append(class_id)

		# Pack instance masks into an array
		if class_ids:
			mask = np.stack(instance_masks, axis=2).astype(np.bool)
			class_ids = np.array(class_ids, dtype=np.int32)
			return mask, class_ids
		else:
			# Call super class to return an empty mask
			return super(CNNabisDataset, self).load_mask(image_id)

	# The following two functions are from pycocotools with a few changes.

	def annToRLE(self, ann, height, width):
		"""
		Convert annotation which can be polygons, uncompressed RLE to RLE.
		:return: binary mask (numpy 2D array)
		"""
		segm = ann['segmentation']
		if isinstance(segm, list):
			# polygon -- a single object might consist of multiple parts
			# we merge all parts into one mask rle code
			rles = maskUtils.frPyObjects(segm, height, width)
			rle = maskUtils.merge(rles)
		elif isinstance(segm['counts'], list):
			# uncompressed RLE
			rle = maskUtils.frPyObjects(segm, height, width)
		else:
			# rle
			rle = ann['segmentation']
		return rle

	def annToMask(self, ann, height, width):
		"""
		Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
		:return: binary mask (numpy 2D array)
		"""
		rle = self.annToRLE(ann, height, width)
		m = maskUtils.decode(rle)
		return m


############################################################
#  Training
############################################################



if __name__ == '__main__':
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN on MS COCO.')
	# trouble shooting line
	#
	parser.add_argument("command",
						metavar="<command>",
						help="'train' or 'evaluate' on MS COCO")
	parser.add_argument('--dataset', required=True,
						metavar="/path/to/coco/",
						help='Directory of the MS-COCO dataset')
	parser.add_argument('--year', required=False,
						default=DEFAULT_DATASET_YEAR,
						metavar="<year>",
						help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
	parser.add_argument('--model', required=False,
	                    default=COCO_MODEL_PATH,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--limit', required=False,
						default=500,
						metavar="<image count>",
						help='Images to use for evaluation (default=500)')
	parser.add_argument('--download', required=False,
						default=False,
						metavar="<True|False>",
						help='Automatically download and unzip MS-COCO files (default=False)',
						type=bool)
	args = parser.parse_args()
	print("Command: ", args.command)
	print("Model: ", args.model)
	print("Dataset: ", args.dataset)
	print("Year: ", args.year)
	print("Logs: ", args.logs)
	print("Auto Download: ", args.download)


	# Configurations
	if args.command == "train":
		config = CNNabisConfig()
	else:
		# class InferenceConfig(CocoConfig):
		#	 # Set batch size to 1 since we'll be running inference on
		#	 # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		#	 GPU_COUNT = 1
		#	 IMAGES_PER_GPU = 1
		#	 DETECTION_MIN_CONFIDENCE = 0
		#	 # for 4-band images
		#	 # IMAGE_CHANNEL_COUNT = 4
		#	 # adjust the mean pixel value for alpha-band, I just made up 100.0
		#	 # MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 100.0])
		config = CNNabisConfig()
	config.display()


	# Create model
	# trouble shooting
	# class args:
	#	 def __init__(self):
	#		 pass
	# args.command = 'train'
	# args.logs = '/home/robot/Documents/_projects/Cannabis/logs/'
	# args.weights = '/home/robot/Documents/_projects/Cannabis/mask_rcnn_coco.h5'
	# args.download = False
	# args.year = '2016'
	# args.dataset = '/media/robot/Storage/images/NAIP_Imagery'
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								  model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
								  model_dir=args.logs)

	# Select weights file to load
	if args.model.lower() == "coco":
		model_path = COCO_MODEL_PATH
	elif args.model.lower() == "last":
		# Find last trained weights
		model_path = model.find_last()
	elif args.model.lower() == "imagenet":
		# Start from ImageNet trained weights
		model_path = model.get_imagenet_weights()
	else:
		model_path = args.model

	# Load weights
	print("Loading weights ", model_path)
	
	# use following line to train from weights already trained on cannabis.
	#model.load_weights(model_path, by_name=True)
	
	# use following line to train from coco weights file
	# model.load_weights(model_path, by_name=True, exclude=['mrcnn_bbox_fc', 'mrcnn_class_logits', 'mrcnn_mask'])
	
	# use following line to train from coco weights file for a 4-band image for the first time then use the
	# subsequent line.
	#model.load_weights(model_path, by_name=True, exclude=['conv1', 'mrcnn_bbox_fc', 'mrcnn_class_logits', 'mrcnn_mask'])
	model.load_weights(model_path, by_name=True)
	# will have to exclude 'conv1' for training 4-band images

	# Train or evaluate
	if args.command == "train":
		# Training dataset. Use the training set and 35K from the
		# validation set, as as in the Mask RCNN paper.
		dataset_train = CNNabisDataset()
		dataset_train.load_coco(args.dataset, "train", year=args.year)
		dataset_train.prepare()

		# Validation dataset
		dataset_val = CNNabisDataset()
		#val_type = "val" #if args.year in '2016' else "minival"
		val_type = "train"  # if args.year in '2016' else "minival"
		dataset_val.load_coco(args.dataset, val_type, year=args.year)
		dataset_val.prepare()

		# Image Augmentation
		# Right/Left flip 50% of the time
		#augmentation = imgaug.augmenters.Fliplr(0.5)
		# also 
		# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
		augmentation = imgaug.augmenters.Sometimes(0.66, [imgaug.augmenters.Fliplr(0.5), imgaug.augmenters.Flipud(0.5), ])
		#	 iaa.Multiply((0.8, 1.2)),
		#	 iaa.GaussianBlur(sigma=(0.0, 5.0))
		# ])
		# augmentation = imgaug.augmenters.Sometimes(0.5, [
		#			 imgaug.augmenters.Fliplr(0.5),
		#			 imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
		#		 ])


		# *** This training schedule is an example. Update to your needs ***

		# Training - Stage 1
		print("Training network heads")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE,
					epochs=30,
					layers='heads',
					augmentation=augmentation)
		# model.train(..., layers='heads', ...)  # Train heads branches (least memory)
		# model.train(..., layers='3+', ...)	 # Train resnet stage 3 and up
		# model.train(..., layers='4+', ...)	 # Train resnet stage 4 and up
		# model.train(..., layers='all', ...)	# Train all layers (most memory)
		
		# For N-Band images, if you train a subset of layers, remember to exclude (?exclude?) conv1 since it's 
		# initialized to random weights. This is relevant if you pass layers="head" 
		# or layers="4+", ...etc. when you call train(). If you are training all layers, skip step
					
		#
		# # Training - Stage 2
		# # Finetune layers from ResNet stage 3 and up
		print("Fine tune Resnet stage 4 and up")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE/10,
					epochs=55,
					layers='3+',
					augmentation=augmentation)
		
		# # Training - Stage 3
		# # Fine tune layers from TesNet stage 4 and up
		print("Fine tune all layers")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 100,
					epochs=75,
					layers='4+',
					augmentation=augmentation)

		# # Training - Stage 4
		# # Fine tune layers from TesNet stage 4 and up
		# print("Fine tune all layers")
		# model.train(dataset_train, dataset_val,
		# 			learning_rate=config.LEARNING_RATE / 5,
		# 			epochs=75,
		# 			layers='all',
		# 			augmentation=augmentation)
