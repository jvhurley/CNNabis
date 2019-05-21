# import the necessary packages
#from imutils import paths
import numpy as np
import cv2
import os
import json
import argparse
from datetime import datetime


# '/home/robot/Documents/_projects/Cannabis/QAQC_annotations/Merged_annotations_2019-05-16_1557990000.json'
# use line below for PyCharm parameter for home computer
# -a=/home/greenbean/Documents/_projects/Cannabis/QAQC_annotations/Merged_annotations_2019-05-16_1557990000.json
# saved in debug parameters

def read_ann_file(annotation_file):
	# this tool will take the path and filename of a COCO -styled annotation file and return
	# three dictionaries: images and cats_by_id. The images dictionary can be called by providing the image
	# name, as in: images['filename.jpg']. Each image will have its own dictionary of annotations, callable by
	# the unique numeric annotations id # starting with 1, as in: images['filename.jpg']['annotations'][1]
	# cats_by_id is a dictionary that will return the category name if given the numeric key, as in:
	# cats_by_id[1] will return the string "outdoor_cannabis_grow" etc
	with open(annotation_file, 'r') as T:
		ann_data = T.read()
	ann_data = json.loads(ann_data)
	images_dict = {}
	images_by_id = {}
	for im in ann_data['images']:
		images_dict[im['file_name']] = im
		images_by_id[im['id']] = im
	for ann in ann_data['annotations']:
		im_id = ann['image_id']
		file_name = images_by_id[im_id]['file_name']
		if 'annotations' not in images_dict[file_name]:
			images_dict[file_name]['annotations'] = {}
		images_dict[file_name]['annotations'][ann['id']] = ann
	categories_by_id = {}
	for cat in ann_data['categories']:
		categories_by_id[cat['id']] = cat['name']
	return images_dict, categories_by_id

def natural_bounds_and_padded(numpy_image, annotation_instance, img_H, img_W, padding=100, scale=600):
	# bounds is used to show the surrounding area around the object of interest. often the objects are too small
	# and if we just showed the object it would be too distorted. Bounds will add 100 pixels around the object
	# scale is the resizing value. Typically a 30 pixel object will have 100 bounds on each side, for a 230 pixel
	# object. this will be resized to 600 in both the vertical and horizontal axes. This operation will not
	# preserve the image relation and may skew the image. 
	x, y, w, h = [int(a) for a in annotation_instance['bbox']]
	if x > padding:
		padded_x = (x - padding)
		x_offset = 0
	else:
		padded_x = 0
		x_offset = x - padding
	if y > padding:
		padded_y = (y - padding)
		y_offset = 0
	else:
		padded_y = 0
		y_offset = y - padding
	if (x + w + (padding * 2)) < img_W:
		padded_w = w + (padding * 2)
		w_offset = 0
	else:
		padded_w = img_W
		w_offset = img_W - (x + w + (padding * 2))
	if (y + h + (padding * 2)) < img_H:
		padded_h = h + (padding * 2)
		h_offset = 0
	else:
		padded_h = img_H
		h_offset = img_H - (y + h + (padding * 2))

	# use the padding to select ROI
	cp_image = numpy_image[padded_y:(padded_y + padded_h), padded_x:(padded_x + padded_w)]
	# get scaling ratios

	cp_image_h_ratio, cp_image_w_ratio, _ = [scale / x for x in cp_image.shape]
	# resize image
	cp_image = cv2.resize(cp_image, (scale, scale))

	# This step take the annotations instance and splits the x and y coordinates
	# into even and odd and subtracts the padding for each. It then creates an empty
	# list to hold the even and odd. It then multiplies each x and y by the scaling ratio
	# and places each in the corresponding and alternating location in the empty list.
	# Then we need to create an numpy array from list and reshape it with (-1,1,2) into
	# a list of coordinates. The -1,1,2 reshape is tricky because it means anylength, 1 column, 2 columns
	even = [frame - padded_x for frame in annotation_instance['segmentation'][0][0:][::2]]
	odd = [frame - padded_y for frame in annotation_instance['segmentation'][0][1:][::2]]
	reproj_seg = [None] * (len(even) + len(odd))
	reproj_seg[::2] = [int(cp_image_w_ratio * i) for i in even]
	reproj_seg[1::2] = [int(cp_image_h_ratio * i) for i in odd]
	reproj_seg = np.array(reproj_seg, np.int32).reshape((-1, 1, 2))
	return cp_image, reproj_seg

if __name__ == "__main__":

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-w", "--working_dir", required=False,
	                default=os.path.dirname(os.path.realpath(__file__)),
	                help='output location, script location will be used if not specified')
	ap.add_argument("-i", "--image_dir", required=True,
	                help="Path to the input images")
	ap.add_argument("-a", "--ann_file", required=True,
	                help="Path and file name of input annotations file")
	args = vars(ap.parse_args())
	# take args and pass to variables
	working_dir = args['working_dir']
	annotation_file = args['ann_file']
	images_dir = args['image_dir']

	images, cats_by_id = read_ann_file(annotation_file)

	# now with an image name, we can loop through each annotation simply
	# by calling its key. This will allow us to open an image once and
	# work with all of the annotations for that image.
	#   images holds all of the necessary info

	# open an image
	# for image in images:
	# file = 'Trinity_019-610-39-00_1A161158CTRI_96041.jpg'
	# file = '427197.jpg'
	file = "428312.jpg"
	img_H, img_W = images[file]['height'], images[file]['width']
	img_path = os.path.join(images_dir, file)
	image = cv2.imread(img_path)

	for count,ann_instance in enumerate(images[file]['annotations'].values()):
		# get the annotation class
		ann_instance_class = cats_by_id[ann_instance['category_id']]
		class_label = f"{ann_instance_class}"
		# we need to create a copy of image otherwise, our annotations will be displayed on each subsequent
		# cp_image
		cp_whole_image = image.copy()
		cp_image, reproj_seg = natural_bounds_and_padded(image, ann_instance, img_H, img_W, padding=100, scale=600)
		text_location = reproj_seg[0][0].copy()
		try:
			text_location[1] = text_location[1] + 50
			text_location[0] = text_location[0] - 150
			text_location = tuple(text_location)
		except:
			text_location[1] = text_location[1] + text_location[1]/2
			text_location[0] = text_location[0] - text_location[1]/2
			text_location = tuple(text_location)
		# image handling details
		cv2.polylines(cp_image, [reproj_seg], True, (0, 0, 255), thickness=1)
		cv2.putText(cp_image, class_label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 9)
		cv2.putText(cp_image, class_label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		
		all_anns = [ann['segmentation'][0] for ann in images[file]['annotations'].values()]
		cv2.polylines(cp_whole_image, [np.array(ann, np.int32).reshape((-1, 1, 2)) for ann in all_anns],
		              True, (0, 0, 255), thickness=2)

		cv2.imshow("Whole_Image", cp_whole_image)
		cv2.imshow("Image", cp_image)
		k = cv2.waitKey(0)
		# break out of all QA processes
		if k == ord('q'):
			cv2.destroyAllWindows()
			break
		# if the annotation is good, overwrite the verified with a timestamp
		if k == ord("s"):
			now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
			images[file]['annotations'][count + 1]['verified'] = now
			# break
		# if the annotation is too bad to fix, write FAIL to verified
		if k == ord("f"):
			images[file]['annotations'][count + 1]['verified'] = "FAIL"
			# break
	cv2.destroyWindow('Image')
	text = 'Is this image completely annotated? (y/n)'
	cv2.putText(cp_whole_image, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,0), 9)
	cv2.putText(cp_whole_image, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	cv2.imshow("Whole_Image", cp_whole_image)
	k = cv2.waitKey(0)
	while True:
		# ask if there are any segmentations that are not annotated?
		if k == ord("y"):
			# this image is good to go.
			break
		elif k == ord("n"):
			# k = cv2.waitKey(0)
			# place tool to make image list for new annotations
			break
		elif k == ord('q'):
			cv2.destroyAllWindows()
			# break
		else:
			k = cv2.waitKey(0)
	cv2.destroyWindow('Whole_Image')
