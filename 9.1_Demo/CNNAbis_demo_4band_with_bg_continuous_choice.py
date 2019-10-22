#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[2]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time
import cv2
import glob
import h5py


# Root directory of the project
ROOT_DIR = os.path.abspath("/home/j5/_projects/Cannabis")

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize2
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco

#get_ipython().magic('matplotlib inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


#  Eager execution switch
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join('/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/logs/cnnabis_naip_2016_with_bg_partial20190912T1750/mask_rcnn_cnnabis_naip_2016_with_bg_partial_0033.h5')
COCO_MODEL_PATH = os.path.join('/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/logs/cnnabis_naip_2016_400bg_minified20191003T1710/mask_rcnn_cnnabis_naip_2016_400bg_minified_0024.h5')
#COCO_MODEL_PATH = os.path.join('/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/logs/cnnabis_naip_2016_400bg_minified20191003T1710/mask_rcnn_cnnabis_naip_2016_400bg_minified_0047.h5')
#COCO_MODEL_PATH = os.path.join('/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/temp_4band_working_no_BG.h5') # this doesn't work with the bg trained examples... You'll have to change number of class and the names of classes
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	#utils.download_trained_weights(COCO_MODEL_PATH)
	print("Could not find model")
	assert os.path.exists(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = '/media/j5/5TB/NAIP_Imagery/2016_NAIP_CNNabis_4band/train2016'
# IMAGE_DIR = '/home/j5/_projects/Cannabis/3.2_background_examples/all_chopped_tifs' # this is a bunch of background images. Make sure it doesn't predict everyting everywhere.
#IMAGE_DIR = os.path.join('/media/j5/NAIP_Imagery/old_stuff/2016_1024x_CNNabis/train2016')

# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[67]:


class CNNabisConfig(Config):
	"""Configuration for inference on MS COCO.
	Derives from the base Config class and overrides values specific
	to the COCO dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "CNNabis_inference"
	
	# Uncomment to train on 8 GPUs (default is 1)
	# GPU_COUNT = 8
	IMAGES_PER_GPU = 1

	# Number of classes (including background)
	NUM_CLASSES = 1 + 7  # CNNabis has 6 classes + 1 back ground
	# NUM_CLASSES = 1 + 6  # CNNabis has 6 classes 

	# Use smaller anchors because our image and objects are small
	#RPN_ANCHOR_SCALES = ( 8, 16, 32, 64, 96, 128)  # failed?!
	#RPN_ANCHOR_SCALES = ( 16, 64, 128, 256, 312)  # anchor side in pixels
	#RPN_ANCHOR_SCALES = ( 32, 64, 128, 256, 384)  # anchor side in pixels
	#RPN_ANCHOR_SCALES = ( 32, 48, 64, 128, 256)  # not small enough to big enough
	RPN_ANCHOR_SCALES = (32, 64, 96, 128, 256)  # The RPN ratios allow us to capture smaller objects, this setting seems to be a good fit for the imagery, (ponds, and outdoor grows at least)
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
config.IMAGES_PER_GPU = 5
config.BATCH_SIZE = config.IMAGES_PER_GPU
config.IMAGE_SHAPE = np.array([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, config.IMAGE_CHANNEL_COUNT])
config.display()


# ## Create Model and Load Trained Weights

# In[68]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[69]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# We'll have to figure out the right order... it may come from train or val... currently these don't match :(
class_names = ['BG', 'outdoor_cannabis', "hoop_greenhouse", "framed_greenhouse", "greenhouse_footprint", "pond", "water_tank", "bg"]
# class_names = ['BG', 'outdoor_cannabis', "hoop_greenhouse", "framed_greenhouse", "greenhouse_footprint", "pond", "water_tank"] # use this for the temp model


# ## Run Object Detection

# In[70]:


# Load a random image from the images folder
file_names = glob.glob(os.path.join(IMAGE_DIR, '*[!_jpg].tif'))
#file_names = next(os.walk(IMAGE_DIR))[2]

def Inference_now(file_names, IMAGE_DIR, model, ):
	imgs_for_inference = []
	titles = []

	for img in range(config.IMAGES_PER_GPU):
		img_choice = random.choice(file_names)
		imgs_for_inference.append(skimage.io.imread(os.path.join(IMAGE_DIR, img_choice )))
		#imgs_for_inference.append(cv2.imread(os.path.join(IMAGE_DIR, img_choice), cv2.IMREAD_UNCHANGED))
		titles.append(img_choice)
		print(img_choice)
	results = model.detect(imgs_for_inference, verbose=1)
	# Visualize results
	for count,rez in enumerate(results):
		##############  Pickel rez for demo or experimentation purposes
		import pickle
		direct = os.getcwd()
		with open(os.path.join(direct, str(count) + '.pickle'), 'wb') as handle:
			pickle.dump(rez, handle, protocol=pickle.HIGHEST_PROTOCOL)
		##############  Pickel rez for demo or experimentation purposes
		# reorder bands
		color_img = imgs_for_inference[count][:,:,:3].copy()
		# we need to reverse the color order for the visualize tool because it is based on skimage and we read in via cv2???
		false_color_img = color_img.copy()
		false_color_img[:,:,0] = imgs_for_inference[count][:,:,3]
		false_color_img[:, :,1] = imgs_for_inference[count][:,:,0]
		false_color_img[:, :,2] = imgs_for_inference[count][:,:,1]
		visualize2.display_instances(color_img, rez['rois'], rez['masks'], rez['class_ids'],
				class_names, rez['scores'], show_mask=False, title=titles[count], false_image=false_color_img)

state = ''
while state != 'n':
	Inference_now(file_names, IMAGE_DIR, model)
	state = input("Continue? (y/n)")
	