# TODO Author, Description, in & outs
# also mention the template.jsonOriginal and how to add images to this file.

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
	# read in template annotation file
	template = '/home/robot/Documents/_projects/Cannabis/staff_annotations/template_noAnns.jsonOriginal'
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

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-a", "--ann_folder", required=True,
	                help="Path of folder containing all annotations files to merge")
	args = vars(ap.parse_args())

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
