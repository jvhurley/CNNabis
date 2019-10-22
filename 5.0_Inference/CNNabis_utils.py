import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from pyproj import Proj, transform
import fiona
from fiona.crs import from_epsg
import os
from multiprocessing import Pool
from itertools import product
from functools import partial
#image = '/home/j5/_projects/Cannabis/Inference/m_4012206_se_10_h_20160717.tif'


def load_img_container(image):
	with rasterio.open(image) as src:
		profile = src.profile
		#meta = src.meta
	num_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
	h, w, d = num_img.shape
	list_of_starting_coords = []
	lim_h = h // 1024
	lim_w = w // 1024
	for row in range(lim_h):
		for col in range(lim_w):
			list_of_starting_coords.append((row * 1024, col * 1024))
		list_of_starting_coords.append((row * 1024, w - 1024))
	for col in range(lim_w):
		list_of_starting_coords.append((h - 1024, col * 1024))
	list_of_starting_coords.append((h - 1024, w - 1024))
	# create numpy array to hold inference data
	# please note that rasterio writes out raster with dimension first!
	s = (7, h, w)
	inference_container = np.zeros(s, dtype=np.uint8)
	# create container for inference
	# Write the product as a raster band to a new 8-bit file. For
	# the new file's profile, we start with the meta attributes of
	# the source file, but then change the band count to 6, set the
	# dtype to uint8, and specify LZW compression.
	profile.update(dtype=rasterio.uint8, count=1)
	# with rasterio.open('example-total.tif', 'w', **profile) as dst:
	#	dst.write(inference_container)
	return num_img, profile, list_of_starting_coords, inference_container


def take_coords_return_coords(original, destination, shapes):
	from pyproj import transform
	new_poly = [[]]
	#print(shapes)
	for coords in shapes:
		for coord in coords:
			#print(len(coords))
			#print(coord)
			x, y = coord
			# tranform the coord
			new_x, new_y = transform(original, destination, x, y)
			# put the coord into a list structure
			poly_coord = tuple([float(new_y), float(new_x)])
			# append the coords to the polygon list
			new_poly[0].append(poly_coord)
	return new_poly


def canna_transform(shapes_in, profile):
	from pyproj import Proj
	# we are going to convert everything to WGS 84 epsg 4326
	crs = profile['crs']
	dst = 'epsg:4326'
	destination = Proj(dst)
	original = Proj(crs)
	#print(shapes_in)
	############   non-parrallelized
	shapes_in['geometry']["coordinates"] = take_coords_return_coords(original,
	                                                                 destination, shapes_in['geometry']['coordinates'])
	return shapes_in
	############  parrallelize attempt 1 this one works!!!
	# polygons = []
	# with Pool(processes=36) as pool:
	# 	polygons.append([x for x in pool.starmap_async(
	# 		take_coords_return_coords,
	# 		product([original], [destination], shapes_in['coordinates'])).get()])
	# shapes_in["coordinates"] = polygons[0]
	# return shapes_in
	############  parrallelize attempt 2
	# polygon = []
	# #func_list = [(shapes2, original, destination) for shapes2 in shapes_in['coordinates']]
	# #print(func_list)
	# func_list = partial(take_coords_return_coords, original, destination)
	# #print(func_list)
	# with Pool(processes=36) as pool:
	# 	polygons.append([x for x in pool.imap_unordered(
	# 		func_list, shapes_in['coordinates'])])
	# shapes_in["coordinates"] = polygons[0]
	#print(polygons[0])
	#return shapes_in


def update_shapefile(num_img, profile):
	with rasterio.Env():
		#print('starting shp_poly creation')
		############  parrallelization attempt
		shp_poly_all = []
		#shapes_in = [x[0] for x in shapes(num_img, mask=None, transform=profile['transform'])]

		all_shapes = [{"properties": {"model_conf": v}, "geometry": s} for i, (s, v) in
		              enumerate(shapes(num_img, mask=None, transform=profile['transform'])) if v != 0]
		# print(all_shapes)
		# print(type(all_shapes))
		# print(all_shapes[0])
		with Pool(processes=35) as pool:
			shp_poly_all.append(pool.starmap(canna_transform, product(all_shapes, [profile])))

		############  working version
		# shp_poly = [{"properties": {"model_conf": v},
		#                    "geometry": canna_transform(crs, dst, s)} for i, (s, v) in
		#                   enumerate(shapes(num_img, mask=None, transform=profile['transform'])) if v != 0]
		#print('finished polygon creation ')
	return shp_poly_all







