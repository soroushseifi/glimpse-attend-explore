# Glimpse-Attend-and-Explore
This repository contains the Pytorch implementaion for the paper "Glimpse-Attend-and-Explore: Self-Attention for Active Visual Exploration":
https://arxiv.org/abs/2108.11717


The code is currently under developement and limited to the recontsruction task but segmentation and classification support will be added soon.

To run the code:

1-Put your dataset in a directory.

2-Save a text file holding the address to the images in the dataset.

3-Edit the 'directory' and 'file' variables in the get_data() function in utils.py to point to the text file.

4-The method would automatically take the last 10% of the files for validation.

5-run the main method.
