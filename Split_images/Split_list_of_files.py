import random
import os
from shutil import copyfile

os.chdir('/media/robot/Storage/images/NAIP_Imagery/2016_1024x_CNNabis')
files = os.listdir()

FF = [file for file in os.listdir() if os.path.splitext(file)[1] == '.JPEG']

random.shuffle(FF)

train, val = FF[:int(1132 * 0.75)],FF[-int(1132 * 0.25):]
train_location = '/media/robot/Storage/images/NAIP_Imagery/2016_1024x_CNNabis/train2016'
val_location ='/media/robot/Storage/images/NAIP_Imagery/2016_1024x_CNNabis/val2016'
assert os.path.isdir(train_location)
assert os.path.isdir(val_location)

for f in train:
	file = os.path.join(train_location, f)
	if not os.path.isfile(file):
		copyfile(f, file)

for f in val:
	file = os.path.join(val_location, f)
	if not os.path.isfile(file):
		copyfile(f, file)