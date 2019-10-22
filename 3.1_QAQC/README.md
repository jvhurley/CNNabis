Author:
	Andrew Dix Hill; https://github.com/AndrewDixHill/CNNabis/QAQC_annotations ; andrew.hill@waterboards.ca.gov

Agency:
	California State Water Resource Control Board (SWRCB)

Purpose:
	A tool for reviewing annotations of images.

How to use this script:
	use the -a= and -i= flags followed by the appropriate arguements. For instance the line below is how I would run
	this code from command line of my home directory, it should be on one line only:
python /full/path/to/Display_image_annotations_withKeys.py -a=/home/robot/Documents/_projects/Cannabis/staff_annotations/annotations/SR_16_thru_30.json -i=/media/robot/Storage/images/NAIP_Imagery/2016_StateRegionalNAIP_examples
Prerequisites:
	Anaconda built with python 3 ; or python 3
	python tools; use "pip install *tool_name*" to install each of these tools (replace *tool_name* with each of the
	tools below):
		numpy
		pymsgbox
		opencv-python