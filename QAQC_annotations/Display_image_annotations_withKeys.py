'''

Author:
	Andrew Dix Hill; https://github.com/AndrewDHill/CNNabis/QAQC_annotations ; andrew.hill@waterboards.ca.gov

Agency:
	California State Water Resource Control Board (SWRCB)

Purpose:
	A tool for reviewing annotations of images.

How to use this script:
	use the -a= and -i= flags followed by the appropriate arguments. For instance the line below is how I would run
	this code from command line of my home directory, it should be on one line only:
		python /full/path/to/Display_image_annotations_withKeys.py -a=/home/robot/Documents/_projects/Cannabis/staff_annotations
		/annotations/SR_16_thru_30.json -i=/media/robot/Storage/images/NAIP_Imagery/2016_StateRegionalNAIP_examples
	the -a flag should specify the location and name of the annotation file
	the -i flag should specify the folder that immediately holds all of the images in your annotation file

Prerequisites:
	Anaconda built with python 3 ; or python 3
	python tools; use "pip install *tool_name*" to install each of these tools (replace *tool_name* with each of the
	tools below):
		numpy
		pymsgbox
		opencv-python
Outputs:
	1. A json file with all of the annotations that passed the QA process named after the input annotation file with
	"_QA_passed" appended to the end of the filename.
	2. A json file with all of the annotations that DID NOT pass the QA process named after the input annotation file
	with "_QA_to_modify" appended to the end of the filename.
	3. A txt file with "_images_for_review" appended to the annotation filename. This may not be a necessary output.
	4. A txt file with "_processed_images" appended to the annotation filename. This is used to keep track of the QA
	review process through each annotation file. Do not modify this file unless you understand why this is important.
	5. A folder of all of the images that will be used to manually adjust polygons or completely annotating an image.

Next steps:
	Once the first QA pass is completed on an input annotation file, a user should use an annotation program like
	ImgLab and open the new annotation file ending with "_QA_to_modify.json". In ImgLab they should use the folder of
	images named after the input annotation file and review each image and annotation for completely labeled images
	or correctly annotated objects. Frequent problems are:
		1. poorly segmented objects. polygons must follow boundaries of objects. This may mean that a user has to
		zoom in fairly closely to correctly annotate an object. If you open an image and only see one or a few
		annotations but there are quite a few unannotated objects, it is likely that only these annotations need
		review. If you do annotated everything in the image, this will not disrupt the training process but it may
		duplicate effort.
		2. An image is missing all annotations of the target categories. If you open an image and see many
		annotations, the review may be because there are missing objects that should be annotated. Please also review
		each annotation for correct segmenation.

	The QA process is an iterative process and should continue until the annotation file ending with
	"_QA_to_modify.json" does not contains any annotations. At this point, all of the files ending with
	"_QA_passed.json" should be merged into a single annotation file with the Merge_annotations.py tool located here:
		https://github.com/AndrewDHill/CNNabis/tree/master/staff_annotations/Merge_annotations.py

'''

# import packages
import numpy as np
import cv2
import os
import json
import argparse
from datetime import datetime
import pymsgbox
import copy
from shutil import copyfile


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
	# a bonifide copy is needed or we just duplicate data in two locations. This blew my mind in how tricky this
	# problem was to track down. Literally, you think you are making a new object but you aren't unless you make a
	# completely different copy!!!!!!!!
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
	return images_d, categories_by_id , ann_data['categories'], ann_data2['images']

def append_anns(json_in, anns_list):
	if os.path.exists(json_in):
		with open(json_in) as anns_in:
			ann_data_in = anns_in.read()
		try:
			ann_data_in = json.loads(ann_data_in)
			for ann in ann_data_in['annotations']:
				anns_list.append(ann)
		except:
			pass
	else:
		with open(json_in, 'w') as open_file:
			open_file.write('')

def natural_bounds_and_padded(numpy_image, annotation_instance, img_H, img_W, padding=100, scale=600):
	# bounds is used to show the surrounding area around the object of interest. often the objects are too small
	# and if we just showed the object it would be too distorted. Bounds will add 100 pixels around the object
	# scale is the resizing value. Typically a 30 pixel object will have 100 bounds on each side, for a 230 pixel
	# object. this will be resized to 600 in both the vertical and horizontal axes. This operation will not
	# preserve the image relation and may skew the image. 
	x, y, w, h = [int(a) for a in annotation_instance['bbox']]

	padded_x = (x - padding) if x > padding else 0
	padded_y = (y - padding) if y > padding else 0
	padded_w = x + w + (padding * 2) if (x + w + (padding * 2)) < img_W else img_W
	padded_h = y + h + (padding * 2) if (y + h + (padding * 2)) < img_H else img_H

	# use the padding to select ROI
	cp_image = numpy_image[padded_y:padded_h, padded_x: padded_w]
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

def CV2_wait_annotation(image_in, cats_by_id, annotation, images_to_review, file_name, named_window="Image"):
	# this is where we are showing an image and waiting for key input. the switch parameter will be passed to the
	# annotation writing tool. Switch of 0 is good, 1 is needs review of polygon, a empty string switch is fail and
	# remove.
	switch=None
	cv2.imshow(named_window, image_in)
	#cv2.moveWindow(named_window, 0,0)
	k = cv2.waitKey(0)

	# break out of all QA processes
	if k == ord('Q'):
		# leave the broken line broken. it gets us out of the loop till we find a better way.
		cv2.destroyAllWindows(named_window)

	# if the annotation is too bad to fix, write FAIL to verified
	elif k == ord("X"):
		annotation['verified'] = 'deleted'
		switch = None
		cv2.destroyWindow(named_window)

	# if the annotation polygon needs to be modified,
	elif k == ord("p"):
		annotation['verified'] = 'needs_review'
		if file_name not in images_to_review:
			images_to_review.append(file_name)
		# if we alter the annotated polygon to be a bounding box, then it will be easier for a use to see which
		# polygons need fixing. This probably isn't the best way but it seems like it could be a workaround until we
		# can color code the polygons.
		bbox = annotation['bbox']
		#print(bbox)
		annotation['segmentation'][0] = [bbox[0],bbox[1], bbox[0],bbox[3], bbox[2],bbox[3], bbox[2], bbox[1], ]
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
		CV2_wait_annotation(image_in, cats_by_id, annotation, images_to_review, file_name, named_window)
	return annotation, switch

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image_dir", required=True,
	                help="Path to the input images")
	ap.add_argument("-a", "--ann_file", required=True,
	                help="Path and file name of input annotations file")
	args = vars(ap.parse_args())

	# take args and pass to variables
	annotation_file = args['ann_file']
	images_dir = args['image_dir']

	# initialize the lists to hold passed anns, anns for review, images needed for review, and processed images
	anns_passed = []
	anns_to_modify = []
	images_to_review = []
	processed_images = []

	# take the annotation file name and create filenames: annotations that passed QA, annotations that need more
	# review, and a list of images that have been processed, and a folder of all the images that will be needed for
	# ImgLab along with the QA_to_modify.json file.
	ann_parts = os.path.splitext(annotation_file)[0]
	processed_images_path = ann_parts + '_processed_images.txt'
	QA_passed = ann_parts + '_QA_passed.json'
	QA_needs_review = ann_parts + '_QA_to_modify.json'
	images_for_review_location = ann_parts

	# check to see if above files and folders exist and if they do, read in their information and append to the
	# specified list, if they do not, create them
	append_anns(QA_passed, anns_passed)
	append_anns(QA_needs_review, anns_to_modify)

	# check if the processed images file exists or open file and append processed image names to our progress list
	if os.path.exists(processed_images_path):
		with open(processed_images_path) as process_images:
			for line in process_images:
				processed_images.extend([line.strip()])
	else:
		with open(processed_images_path, 'w') as open_file:
			open_file.write('')
	# check if images for review folder exists
	if not os.path.exists(images_for_review_location):
		os.mkdir(images_for_review_location)

	# call the read_ann_file definition. This creates a dictionary of image name keys that return a dictionary of
	# atributes: file_name, height, width, id, and all of the associated annotations, If these annotations have not
	# been through the QA process yet, they should not have a verified key.
	# images  -----------  used to loop over each image. It also holds all of the annotations for each image.
	# cats_by_id  ------- used to convert numeric category ids to strings
	#
	# cats_dict  -------  a list of category dictionaries, used to construct category dictionary for the output json files
	# images_dict   ------- this is used to construct the image dictionary for the output json files
	#
	# The images and cats_by_id will be used to collect and call information.
	images, cats_by_id, cats_dict, images_dict = read_ann_file(annotation_file)

	# with an image name, we can loop through each annotation simply
	# by calling its key. This will allow us to open an image once and
	# work with all of the annotations for that image.
	for file in images:
		# it would be better to used a named value in a dictionary instead of looping over the entire list. I believe
		# that images_dict could do this
		if 'annotations' not in images[file]:
			continue
		if file =='delete.JPEG':
			for count, ann_instance in enumerate(images[file]['annotations'].values()):
				if ann_instance not in anns_to_modify:
					anns_to_modify.append(ann_instance)
			continue
		# check if this file is in the processed list already
		if file in processed_images:
			# this is how we skip images that have already been processed.
			continue
		# capture the height and width of image
		img_H, img_W = images[file]['height'], images[file]['width']
		# capture the image path and read it into a numpy array
		img_path = os.path.join(images_dir, file)
		image = cv2.imread(img_path)
		# we need to create a copy of image otherwise, any edits to the image will be displayed each time we want to
		# show it.
		cp_whole_image = image.copy()

		# make a list of all the annotations dictionaries.
		all_anns = [ann['segmentation'][0] for ann in images[file]['annotations'].values()]

		# display the whole image with all of its annotations. If there is missing annotations the whole image and
		# all of its annotations will be sent for review. A user will see the whole image with all of the annotations
		# and will be asked if we annotated everything after reviewing all of the annotations
		cv2.polylines(cp_whole_image, [np.array(ann, np.int32).reshape((-1, 1, 2)) for ann in all_anns],
		              True, (0, 0, 255), thickness=2)
		# display the image and move it 650 pixels to the right to accommodate the annotation review panel.
		cv2.imshow("Whole_Image", cp_whole_image)
		cv2.moveWindow("Whole_Image", 650, 0)

		# loop over each annotation for an image
		# Create a variable to track ho wmany annotations need review
		annotation_number_needing_review = 0
		for count,ann_instance in enumerate(images[file]['annotations'].values()):
			# if the verified key exists in an annotation, it has been through this QA process at least once.
			# check to see if it has a timestamp in the verified attribute and skip it is if does, otherwise continue
			# with annotation review. Converting to a time stamp and throwing an error to check if it matches a
			# pattern... is probably not the most efficient way to do this!
			if 'verified' in ann_instance:
				QA_check = ann_instance['verified']
				try:
					datetime.strptime(QA_check, "%Y-%m-%d_%H:%M:%S")
					continue
				except:
					pass

			# get the annotation class using the number from the annotation
			class_label = f"{cats_by_id[ann_instance['category_id']]}"

			# show instructions on separate window. Include category name, instructions and the filename.
			instructions_img = np.zeros(shape=(350, 650, 3)).astype('uint8')
			QA_text = f'class: {class_label}\nQ: quit\nX: delete annotation\np: send to review for polygon change\nc: change ' \
				f'category and ' \
				f'advance\ns: save annotation and advance\nfile: {file}'
			y0, dy = 50, 40
			for i, line in enumerate(QA_text.split('\n')):
				y = y0 + i * dy
				cv2.putText(instructions_img, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
				cv2.putText(instructions_img, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.imshow("instructions", instructions_img)
			cv2.moveWindow("instructions", 0, 700)

			# use the natural_bounds_and_padded tool to crop the annotation with a specified padding from the bbox
			# around the annotation and scale the image and annotation instance to the scale value.
			# cp_image is the nummpy arrary of the croped image
			# reproj_seg is the reprojected annotation segmentation scale to fit the crop and rescale cp_image
			cp_image, reproj_seg = natural_bounds_and_padded(image, ann_instance, img_H, img_W, padding=100, scale=600)
			# make a copy of the location of the reprojection segmenation in an effort to place category information
			# near the segmenation. Specifically below the instance. If the text is placed outside of the image,
			# cv2 already handles the errors.
			text_location = reproj_seg[0][0].copy()
			text_location[1] = text_location[1] + 100
			text_location[0] = text_location[0] - 150
			text_location = tuple(text_location)

			# write annotation segmenation information, and category information to the cropped image.
			cv2.polylines(cp_image, [reproj_seg], True, (0, 0, 255), thickness=1)
			cv2.putText(cp_image, class_label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 9)
			cv2.putText(cp_image, class_label, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

			# the CV2_wait_annotation tool displays the cropped image and waits for key input from use. Based on key
			# input, various actions are taken. ann_reviewed hold the specific annotation and modifications,
			# Switch is used to append this annotation to either the passed annotation list or the needs modification
			# annotation list. If switch is None, user meant to delete the annotation.
			ann_reviewed, switch = CV2_wait_annotation(cp_image, cats_by_id, ann_instance, images_to_review, file,
			                                           named_window="Annotation_review")

			# at the end of annotation review, destroy the instructions window.
			cv2.destroyWindow("instructions")
			# append ann_reviewed using switch to determine correct dictionary
			if switch == 0:
				anns_passed.append(ann_reviewed)
			elif switch == 1:
				if ann_reviewed not in anns_to_modify:
					anns_to_modify.append(ann_reviewed)
				annotation_number_needing_review += 1
				# we need to include all of the annotations
			elif switch == None:
				continue
		# if annotation_number_needing_review is greater than 0 it means at least one annotation needs review and
		# all annotations should be passed to the review process. We are doing this here because the "verified"
		# values may not be present prior to this step.
		if annotation_number_needing_review > 0:
			# pass all annotations to the review process so that user can see what annotations have been done. It
			# will require the annotator to take a close look at each polygon to determine if they need review.
			# Hopefully I can find a way to color code polygons in ImgLab based on the "verified" value.
			for count, ann_instance in enumerate(images[file]['annotations'].values()):
				if ann_instance not in anns_to_modify:
					anns_to_modify.append(ann_instance)

		# display the whole image with all of its annotations. If there is missing annotations the whole image and
		# all of its annotations will be sent for review.
		# if a staff's annotation efforts are going through the first round of annotation efforts, then we should ask
		# them if we've captured all of the target annotation categories within a single image. If we've missed
		# something, we need to send it back for review because the training process may use non-annotations as
		# background training examples.
		# if this is the second or greater round of QA, then we don't need to ask if we've captured all of the
		# annotations because we assume this happened in the first round of QA. We use the filename ending to
		# ascertain whether or not this is the first round of annotations.
		if os.path.splitext(annotation_file)[0].endswith('QA_to_modify'):
			pass
		else:
			text = f'Did we capture all of the target categories? (y/n)'
			cv2.putText(cp_whole_image, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 9)
			cv2.putText(cp_whole_image, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.putText(cp_whole_image, f'{file}', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 9)
			cv2.putText(cp_whole_image, f'{file}', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.imshow("Whole_Image", cp_whole_image)
			#cv2.moveWindow("Whole_Image", 0, 0)
			k = cv2.waitKey(0)
			while True:
				# break out of all QA processes
				if k == ord('Q'):
					cv2.destroyAllWindows()
				# ask if all target categories are captured by annotations in an image?
				if k == ord("y"):
					break
				elif k == ord("n"):
					# we need to pass this image into the folder of images needed for review.
					if file not in images_to_review:
						images_to_review.append(file)
					# pass all annotations to the review process so that user can see what is missing unless they
					# were deleted!
					for count, ann_instance in enumerate(images[file]['annotations'].values()):
						if any([seg['segmentation'] == ann_instance['segmentation'] for seg in anns_to_modify]):
							continue
						# if any ann_instance['segmentation'] == anns_to_modify
						else:
							if ann_instance not in anns_to_modify and ann_instance['verified'] != 'deleted':
								anns_to_modify.append(ann_instance)
							else:
								continue
					break
				elif k == ord('Q'):
					cv2.destroyAllWindows()
					break
				else:
					k = cv2.waitKey(0)
			# if we've made it this far, then we can call this image processed and prevent it from going to review again
			processed_images.append(file)


		# write out annotation files for good and bad annotations
		# use the images_dict, cats_dict, annotations_passed, annotations_need_review
		data_passed = {}
		data_passed['images'] = images_dict
		data_passed['categories'] = cats_dict
		data_to_review = copy.deepcopy(data_passed)
		data_passed['annotations'] = anns_passed
		data_to_review['annotations'] = anns_to_modify

		# we have to be careful here as a code break here could erase our data.
		tmp1 = ann_parts + '.tmp1'
		tmp2 = ann_parts + '.tmp2'
		tmp3 = ann_parts + '.tmp3'
		success = ''
		try:
			with open(tmp1, 'w') as W:
				W.write(str(data_passed).replace('\'', '\"').replace(' ', ''))
			with open(tmp2, 'w') as W:
				W.write(str(data_to_review).replace('\'', '\"').replace(' ', ''))
			with open(tmp3, 'w') as W:
				for item in processed_images:
					W.write(f'{item}\n')
			success = 1
		except:
			# don't overwrite the work we've done thus far
			break
		if success == 1:
			copyfile(tmp1, QA_passed)
			copyfile(tmp2, QA_needs_review)
			copyfile(tmp3, processed_images_path)
			# copy over the files we've completed this session.
			for item in images_to_review:
				copyfile(os.path.join(images_dir, item), os.path.join(images_for_review_location, item))
			os.remove(tmp1)
			os.remove(tmp2)
			os.remove(tmp3)
