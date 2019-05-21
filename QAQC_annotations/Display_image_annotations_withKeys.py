# import the necessary packages
#from imutils import paths
import numpy as np
import cv2
import os
import json
import argparse
from datetime import datetime
import pymsgbox
import copy



# '/home/robot/Documents/_projects/Cannabis/QAQC_annotations/Merged_annotations_2019-05-16_1557990000.json'
# use line below for PyCharm parameter for home computer
# -a=/home/greenbean/Documents/_projects/Cannabis/QAQC_annotations/Merged_annotations_2019-05-16_1557990000.json
# saved in debug parameters

def read_ann_file(annotation_file):
	# this tool will take the path and filename of a COCO -styled annotation file and return
	# three dictionaries: images_d, cats_by_id, categories dict and images dict. The images dictionary can be called by
	# providing the image name, as in: images['filename.jpg']. Each image will have its own dictionary of
	# annotations, callable by the unique numeric annotations id # starting with 1, as in: images['filename.jpg'][
	# 'annotations'][1] cats_by_id is a dictionary that will return the category name if given the numeric key, as in:
	# cats_by_id[1] will return the string "outdoor_cannabis_grow" etc
	with open(annotation_file, 'r') as T:
		ann_data_in = T.read()
	ann_data = json.loads(ann_data_in)
	ann_data2 = copy.deepcopy(ann_data)
	images_d = {}
	images_by_id = {}
	for im in ann_data['images']:
		images_d[im['file_name']] = im
		images_by_id[im['id']] = im
	for ann in ann_data['annotations']:
		im_id = ann['image_id']
		file_name = images_by_id[im_id]['file_name']
		if 'annotations' not in images_d[file_name]:
			images_d[file_name]['annotations'] = {}
		images_d[file_name]['annotations'][ann['id']] = ann
	categories_by_id = {}
	for cat in ann_data['categories']:
		categories_by_id[cat['id']] = cat['name']
	return images_d, categories_by_id , ann_data['categories'], ann_data['images']

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

def CV2_wait_annotation(image_in, cats_by_id, annotation, named_window="Image"):
	# this is where we are showing an image and waiting for key input. the switch parameter will be passed to the
	# annotation writing tool. Switch of 0 is good, 1 is needs review of polygon, a empty string switch is fail and
	# remove.
	switch=None
	cv2.imshow(named_window, image_in)
	k = cv2.waitKey(0)

	# break out of all QA processes
	if k == ord('Q'):
		# leave the broken line broken. it gets us out of the loop till we find a better way.s
		cv2.destroysWindow(named_window)

	# if the annotation is too bad to fix, write FAIL to verified
	elif k == ord("X"):
		annotation['verified'] = 'FAIL'
		switch = None
		cv2.destroyWindow(named_window)

	# if the annotation polygon needs to be modified,
	elif k == ord("p"):
		annotation['verified'] = 'needs review'
		switch = 1
		cv2.destroyWindow(named_window)

	# if the annotation category needs to be modified,
	elif k == ord("c"):
		a = '\n'.join([f'{cat}: {count}' for count,cat in cats_by_id.items()])
		input_cat = ''
		while input_cat not in range(1,len(cats_by_id) + 1):
			input_cat = pymsgbox.prompt(f"enter an Category number:\n{str(a)}", 'Change Category')
			try:
				input_cat = int(input_cat)
			except ValueError:
				input_cat = int(pymsgbox.prompt(f"THE\n******   NUMBER ******\nPLEASE:\n{str(a)}", 'Change Category'))
		annotation['category_id'] = input_cat
		now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
		annotation['verified'] = now
		switch = 0
		# append ann to good list
		cv2.destroyWindow(named_window)

	# if the annotation is good, overwrite the verified with a timestamp
	elif k == ord("s"):
		now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
		annotation['verified'] = now
		switch = 0
		# append ann to good list
		cv2.destroyWindow(named_window)
	else:
		CV2_wait_annotation(image_in, cats_by_id, annotation, named_window="Image")
	return annotation, switch


# def append_annotation(ann, switch, write_now):
# 	if switch == 0:
# 		write_out = f'QA_annotations_{now}.json'
# 	else:
# 		write_out = f'annotations_for_review_{now}.json'


if __name__ == "__main__":
	write_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
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

	# initialize the lists to hold reviewed annotations
	anns_passed = []
	anns_to_modify = []
	# call the read_ann_file definition. The cats_dict and the images_dict will be used to create new annotation
	# files. The images and cats_by_id will be used to collect and call information.
	images, cats_by_id, cats_dict, images_dict = read_ann_file(annotation_file)

	# now with an image name, we can loop through each annotation simply
	# by calling its key. This will allow us to open an image once and
	# work with all of the annotations for that image.
	#   images holds all of the necessary info

	# open an image
	# for image in images:
	# file = 'Trinity_019-610-39-00_1A161158CTRI_96041.jpg'
	# file = '427197.jpg'
	# file = "428312.jpg"
	for file in images:
		#file = file_name['file_name']
		if 'annotations' not in images[file]:
			continue
		if file =='delete.JPEG':
			continue
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
				text_location = tuple(5,25)
			# image display details
			cv2.polylines(cp_image, [reproj_seg], True, (0, 0, 255), thickness=1)
			cv2.putText(cp_image, class_label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 9)
			cv2.putText(cp_image, class_label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

			all_anns = [ann['segmentation'][0] for ann in images[file]['annotations'].values()]
			QA_text = 'Q: quit\nX: delete\np: send to review for polygon change\nc: change category and save\ns: save\n'
			y0, dy = 50, 40
			for i, line in enumerate(QA_text.split('\n')):
				y = y0 + i * dy
				cv2.putText(cp_whole_image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
				cv2.putText(cp_whole_image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
			cv2.polylines(cp_whole_image, [np.array(ann, np.int32).reshape((-1, 1, 2)) for ann in all_anns],
			              True, (0, 0, 255), thickness=2)

			cv2.imshow("Whole_Image", cp_whole_image)
			ann_reviewed, switch = CV2_wait_annotation(cp_image, cats_by_id, ann_instance, named_window="Image")
			# append ann_reviewed using switch to determine correct dictionary
			if switch == 0:
				anns_passed.append(ann_reviewed)
			elif switch == 1:
				anns_to_modify.append(ann_reviewed)
			else:
				continue
		#cv2.destroyWindow('Image')
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

		# write out annotation files for good and bad annotations
		# use the images_dict, cats_dict, annotations_passed, annotations_need_review
		data_passed = {}
		data_passed['images'] = images_dict
		data_passed['categories'] = cats_dict
		data_to_review = data_passed.copy()
		data_passed['annotations'] = anns_passed
		data_to_review['annotations'] = anns_to_modify

		with open(os.path.join(working_dir, 'QA_passed.json'), 'w') as W:
			W.write(str(data_passed).replace('\'', '\"').replace(' ', ''))
		with open(os.path.join(working_dir, 'QA_to_modify.json'), 'w') as W:
			W.write(str(data_to_review).replace('\'', '\"').replace(' ', ''))
