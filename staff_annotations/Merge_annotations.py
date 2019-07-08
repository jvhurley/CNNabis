'''

Author:
	Andrew Dix Hill; https://github.com/AndrewDHill/CNNabis/staff_annotations ; andrew.hill@waterboards.ca.gov

Agency:
	California State Water Resource Control Board (SWRCB)

Purpose:
	A tool for merging annotation files.

How to use this script:
	use the -t= and -i= flags followed by the appropriate arguments. For instance the line below is how I would run
	this code from command line, it should be on one line only:
		python /full/path/to/Merge_annotations.py -t=/full/path/to/template.jsonOriginal
		-i=/full/path/to/folder_containing_all_annotations_json_files_to_be_merged/
	the -t flag should specify the location and name of the template json annotation file. This is important because
	the images should have unique id but may be assigned different ids if the annotations were not started with the
	same starting template. If the annotation files to be merged include images that are not in the template json,
	the script will throw an exception.
	the -i flag should specify the folder that holds all of the json annotation files you wish to merge.

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
import os
import json
import argparse
import glob
from datetime import datetime
from copy import deepcopy


def correct_file(file, images, cats, dict_image_segmentation_count):
	# read in first file and convert... put it in a for loop later
	with open(file, 'r') as T:
		TEMP = T.read()
	data_for_review = json.loads(TEMP)

	# used this one
	images_by_ids_needs_review = {}
	for im in data_for_review['images']:
		images_by_ids_needs_review[im['id']] = im
	cats_by_id_for_review = {}
	for cat in data_for_review['categories']:
		cats_by_id_for_review[cat['id']] = cat
	# convert the annotations to match the template category numbers
	point_seg_count = 0
	duplicate_count = 0
	skip_count = 0
	reviewed_anns = []
	real_ann_count = 0
	for count,ann in enumerate(data_for_review['annotations']):
		# skip all segmentations that are only a point and delete them
		seg = ann['segmentation'][0]
		if len(seg) < 3:
			#print('\n\npoint segmentation ...  removing segmentation')
			# remove item from list
			# del data_for_review['annotations'][count]
			point_seg_count += 1
			# count -= 1
			continue
		# get image number information
		review_img_name = images_by_ids_needs_review[ann['image_id']]['file_name']
		# skip adding the template delete.JPEG annotations !!
		# unless the user didn't use the template and
		if review_img_name == 'delete.JPEG':
			#print('\n\nskipping template delete.JPEG annotations')
			#del data_for_review['annotations'][count]
			skip_count += 1
			# count -= 1
			continue

		# here we are trying to keep track of the annotations per image because they need unique ids!
		# and make sure we aren't adding duplicate to the record
		# seg is the segment instance already
		if review_img_name not in dict_image_segmentation_count:
			seg_num = 1
			dict_image_segmentation_count[review_img_name] = {seg_num:seg}
		elif any(seg == segment for segment in dict_image_segmentation_count[review_img_name].values()):
			#print('skipping duplicate annotation')
			#del data_for_review['annotations'][count]
			duplicate_count += 1
			# count -= 1
			continue
		else:
			seg_num = len(dict_image_segmentation_count[review_img_name]) + 1
			dict_image_segmentation_count[review_img_name][seg_num] = seg
		# get template number from name
		try:
			real_image_number = images[review_img_name]['id']
		except:
			print(f"{review_img_name} was not in original set!!!")
			break
		# get annotation number information
		cat_number = cats_by_id_for_review[ann['category_id']]['name']
		# get real category number
		try:
			real_cat_number = cats[cat_number]['id']
		except:
			print(f"{cats_by_id_for_review[ann['category_id']]}\n check out category info")
			break
		# write corrected values
		# check to see if we've got the correct locations!!
		ZZ = images_by_ids_needs_review[ann['image_id']]['file_name']
		check_ZZ = images_by_ids_needs_review[data_for_review['annotations'][count]['image_id']]['file_name']
		if ZZ == check_ZZ:
			data_for_review['annotations'][count]['image_id'] = real_image_number
			data_for_review['annotations'][count]['category_id'] = real_cat_number
			data_for_review['annotations'][count]['id'] = seg_num
		else:
			print('WTF')

		reviewed_anns.append(data_for_review['annotations'][count])
		real_ann_count += 1
	print(f'There were {count - point_seg_count - duplicate_count - skip_count} segmentations written\n'
	      f'There were {point_seg_count} single point segmentations\n'
	      f'There were {duplicate_count} Duplicates in  .........................{file}\n')
	return reviewed_anns, dict_image_segmentation_count


if __name__ == "__main__":

	now = datetime.strftime(datetime.now().date(), '%Y-%m-%d')

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-a", "--ann_folder", required=True,
	                help="Path of folder containing all annotations files to merge")
	ap.add_argument("-t", "--template_file", required=True,
	                help="Path and name of template annotation file")
	args = vars(ap.parse_args())

	# read in template annotation file
	template = args['template_file']
	with open(template, 'r') as T:
		TEMP = T.read()
	verified_data = json.loads(TEMP)
	data = deepcopy(verified_data)
	data['annotations'] = {}
	images = {}
	for im in verified_data['images']:
		images[im['file_name']] = im
	cats = {}
	for cat in verified_data['categories']:
		cats[cat['name']] = cat

	# take args and pass to variables
	annotation_folder = args['ann_folder']
	os.chdir(annotation_folder)
	outFile = os.path.join(annotation_folder, f'Merged_annotations_{now}.json')

	# get a list of all the json files in this location.
	FLs = glob.glob('*.json')

	# if main clause
	annotations = []
	# we need to keep track of unique identifiers for annotations per image.
	# therefore, we will use the image name as the key and the value will be the number of annotations
	# starting with 1. We will use this number to write the id in the segmentation id field.
	dict_image_segmentation_count = {}


	for file in FLs:
		dat, dict_image_segmentation_count = correct_file(file, images, cats, dict_image_segmentation_count)
		annotations.extend(dat)

	data['annotations'] = annotations

	with open(outFile, 'w') as W:
		W.write(str(data).replace('\'','\"').replace(' ', ''))
