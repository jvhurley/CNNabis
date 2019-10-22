# NAIP_retriever
These scripts are designed to download specified NAIP_2016 imagery using a list (/County_lists/Shasta_selection.csv) of unique identifiers (a txt file with just the unique NAIP DOQQ id separate by newlines).

The NAIP_download.py file will download your list of images in a location you specify, skipping ones that have already been written. If you cancel the process, it will remove any partially downloaded images.
