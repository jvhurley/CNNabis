A collection of tools, workflows, and documentation to build an annotated image dataset for object segmentation based
 on the 
MaskRCNN tool. 

The stages of this tool will include:
 1. Manual annotation using the ImgLab tools and process
 2. QA/QC of annotations
 3. Manual correction of flagged annotations.
 4. CNN training using the MaskRCNN approach based solely on annotations that passed QA/QC
 5. Use the trained CNN to augment the annotation dataset
 6. Re-training the CNN 
 7. Deployment