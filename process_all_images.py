# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:39:02 2021

@author: Alex
"""

from SIMILE import image_processing

SOURCE_DIR = 'C:\\Users\\Alex\\Documents\\data_science\\building-stones-clone\\building-stones\\data\\known'
DEST_DIR = './sample_processed'
X_SIZE = 256
Y_SIZE = 256
TEST_PROP = 0.4
REMOVE_SMALL_REGIONS = True
MIN_REGION_SIZE = 30
ROTATE_IMAGES = True
LOWER_FILTER = None
UPPER_FILTER = None


if __name__ == "__main__":


    image_processing.process_all_images(SOURCE_DIR, 
                       DEST_DIR, X_SIZE, 
                       Y_SIZE, TEST_PROP, 
                       MIN_REGION_SIZE, 
                       ROTATE_IMAGES,
                       'blue',
                       LOWER_FILTER,
                       UPPER_FILTER)
    
    image_processing.process_all_images(SOURCE_DIR, 
                       DEST_DIR, X_SIZE, 
                       Y_SIZE, TEST_PROP, 
                       MIN_REGION_SIZE, 
                       ROTATE_IMAGES,
                       'black',
                       LOWER_FILTER,
                       UPPER_FILTER)
    
    image_processing.merge_layers(DEST_DIR)
    