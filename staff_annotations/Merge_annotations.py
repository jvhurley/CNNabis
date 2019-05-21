# TODO Author, Description, in & outs
# also mention the template.jsonOriginal and how to add images to this file.


import os
import json
import glob
from datetime import datetime


now = datetime.strftime(datetime.now().date(), '%Y-%m-%d_%s')
os.chdir('/home/robot/Documents/_projects/Cannabis/staff_annotations')
home_dir = os.getcwd()
template = os.path.join(home_dir, 'template_noAnns.jsonOriginal')
with open(template, 'r') as T:
	TEMP = T.read()
data = json.loads(TEMP)

# relocate to json files
os.chdir('/home/robot/Documents/_projects/Cannabis/staff_annotations/annotations')
work_dir = os.getcwd()
outFile = os.path.join(home_dir, f'Merged_annotations_{now}.json')

def correct_ZZ(file, data, dict_image_segmentation_count):
	# read in first file and convert... put it in a for loop later
	zz = file
	with open(zz, 'r') as T:
		TEMP = T.read()
	ZZ = json.loads(TEMP)
	images = {}
	for im in data['images']:
		images[im['file_name']] = im
	cats = {}
	for cat in data['categories']:
		cats[cat['name']] = cat
	# used this one
	ZZ_images_by_ids = {}
	for im in ZZ['images']:
		ZZ_images_by_ids[im['id']] = im
	ZZ_cats_by_id = {}
	for cat in ZZ['categories']:
		ZZ_cats_by_id[cat['id']] = cat
	# convert the annotations to match the template category numbers
	point_seg_count = 0
	duplicate_count = 0
	skip_count = 0
	for count,ann in enumerate(ZZ['annotations']):
		# skip all segmentations that are only a point and delete them from ZZ['anns']
		seg = ann['segmentation'][0]
		if len(seg) < 3:
			#print('\n\npoint segmentation ...  removing segmentation')
			# remove item from list
			#print(ZZ['annotations'][count])
			del ZZ['annotations'][count]
			point_seg_count += 1
			continue
		# skip adding the template delete.JPEG annotations !!
		# unless the user didn't use the template and
		if ann['image_id'] == 1 and ZZ_images_by_ids[ann['image_id']]['file_name'] == 'delete.JPEG':
			#print('\n\nskipping template delete.JPEG annotations')
			del ZZ['annotations'][count]
			skip_count += 1
			continue
		# get image number information
		ZZ_img_name = ZZ_images_by_ids[ann['image_id']]['file_name']
		# here we are trying to keep track of the annotations per image because they need unique ids!
		# and make sure we aren't adding duplicate to the record
		# seg is the segment instance already
		if ZZ_img_name not in dict_image_segmentation_count:
			seg_num = 1
			dict_image_segmentation_count[ZZ_img_name] = {seg_num:seg}
		elif any(seg == segment for segment in dict_image_segmentation_count[ZZ_img_name].values()):
			#print('skipping duplicate annotation')
			duplicate_count += 1
			del ZZ['annotations'][count]
			continue
		else:
			seg_num = len(dict_image_segmentation_count[ZZ_img_name]) + 1
			dict_image_segmentation_count[ZZ_img_name][seg_num] = seg
		# get template number from name
		try:
			real_image_number = images[ZZ_img_name]['id']
		except:
			print(f"{ZZ_img_name} was not in original set!!!")
			break
		# get annotation number information
		zz_cat_number = ZZ_cats_by_id[ann['category_id']]['name']
		# get real category number
		try:
			real_cat_number = cats[zz_cat_number]['id']
		except:
			print(f"{ZZ_cats_by_id[ann['category_id']]}\n check out category info")
			break
		# write corrected values
		ZZ['annotations'][count]['image_id'] = real_image_number
		ZZ['annotations'][count]['category_id'] = real_cat_number
		ZZ['annotations'][count]['id'] = seg_num
		ZZ['annotations'][count]['verified'] = ''
		count +=1
	print(f'There were {count - point_seg_count - duplicate_count - skip_count} segmentations written\n'
	      f'There were {point_seg_count} single point segmentations\n'
	      f'There were {duplicate_count} Duplicates in  .........................{file}\n')
	return ZZ['annotations'], dict_image_segmentation_count

# get a list of all the json files in this location.
FLs = glob.glob('*.json')

# if main clause
annotations = []
# we need to keep track of unique identifiers for annotations per image.
# therefore, we will use the image name as the key and the value will be the number of annotations
# starting with 1. We will use this number to write the id in the segmentation id field.
dict_image_segmentation_count = {}


for file in FLs:
	dat, dict_image_segmentation_count = correct_ZZ(file, data, dict_image_segmentation_count)
	annotations.extend(dat)

data['annotations'] = annotations

with open(outFile, 'w') as W:
	W.write(str(data).replace('\'','\"').replace(' ', ''))
