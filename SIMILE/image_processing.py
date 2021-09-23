# -*- coding: utf-8 -*-
"""

"""

import cv2
import os
import random
from PIL import Image
import glob
from skimage import morphology
import numpy as np



def remove_small_regions(image_mask, min_region_area):
    """
    Removes contiguous regions of area below a given number of pixels from
    an image mask
    
        Parameters:
            image_mask (Array): binary array of masked porespace areas
            min_region_area (int): minimum size of regions to pass filter
            
        Returns:
            filtered_img (Array): binary array with threshold areas removed
    
    """

    filtered_img = morphology.remove_small_holes(image_mask, area_threshold = min_region_area)
    return filtered_img



def slice_images(
        source_directory,
        fileName,
        tile_height,
        tile_width):
    """
    Reads in an image file, slices it into tiles and saves tiles to 
    disk with sequential filenames.
    
        Parameters:
            source_directory (str): path to input image
            fileName (str): filename of the image to be sliced
            tile_height (int): desired height of tiles in pixels
            tile_width (int): desired width of tiles in pixels
            
        Returns:
            None
    
    """
    

    if not os.path.exists('sliced_images'):
        os.makedirs('sliced_images')
        
    temp_directory = './sliced_images'
    
    k = 0
    im = Image.open(os.path.join(source_directory, fileName))
    imgwidth = im.size[0]
    imgheight = im.size[1]
    for i in range(0, imgheight, tile_height):
        for j in range(0, imgwidth, tile_width):
            box = (j, i, j + tile_width, i + tile_height)

            try:
                a = im.crop(box)
                if(i + tile_height < imgheight and j + tile_width < imgwidth):

                    a.save(
                        os.path.join(
                            temp_directory,
                            (fileName +
                             "IMG-%s.png" %
                             k)))

            except BaseException:
                pass
            k += 1





# returns the binary image
def convert_to_binary(input_file, 
                      lower_threshold, 
                      upper_threshold, 
                      filter_color = 'blue'):
    """
    Reads in image file and applies color thresholding to produce a color mask.
    
        Parameters:
            input_file (str): path to input image
            lower_threshold(tuple): HSV values to define lower bound of color
            filter
            upper_threshold (tuple): HSV values to define upper bound of color 
            filter
            filter_color (str), optional: option to use predefined filters -
            either blue or black (use custom otherwise)
            
        Returns:
            inv_mask (Array): binary array with color filters applied
    
    """

    
    assert filter_color in ['blue','black','custom'], "filter must be blue \
    black or custom"
    
    input_img = cv2.imread(input_file)

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    
    if(filter_color == 'black'):
    #black mask
        mask = cv2.inRange(hsv, (0, 0, 0), (255, 255, 50))
        
    if(filter_color == 'blue'):
        # blue mask
        mask = cv2.inRange(hsv, (70, 50, 50), (140, 255, 255))
        
    if(filter_color == 'custom'):
        mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    
    
    inv_mask = cv2.bitwise_not(mask)

    return inv_mask




# processes all the raw images in a given directory
def process_all_images(
        source_directory,
        dest_directory,
        tile_width,
        tile_height,
        test_prop,
        min_region_size,
        rotate_tiles,
        filter_color,
        lower_threshold,
        upper_threshold):
    
    """
    Takes a directory of raw images and performs all necessary preprocessing 
    to form a training and testing set for the autoencoder
    
        Parameters:
            source_directory (str): path to input image
            dest_directory (str): directory to save processed images to
            tile_width (int): desired height of sliced images in pixels
            tile_height (int): desired width of sliced images in pixels
            test_prop (float): proportion of images to be used in the test set
            min_region_size (int): minimum pixel area of region to keep in 
            mask (zero to keep all regions)
            rotate_tiles (bool): duplicate training set image by 90 degree 
            rotations (for augmentation)
            filter_color (str): pre-defined filter color to use (blue, black 
                                                                 or None)
            lower_threshold (tuple): HSV value for lower mask filter if 
            no filter_color used
            upper_threshold (tuple): HSV value for upper mask filter if 
            no filter_color used
            
        Returns:
            None
    
    """
    
    random.seed(57) #essential since we call this function once for each layer
    
    assert filter_color in ['blue','black','custom'], "filter must be blue \
    black or custom"
    
    
    if not os.path.exists('sliced_images'):
        os.makedirs('sliced_images')
        
    temp_directory = './sliced_images'
    
    if(filter_color == "blue" or filter_color == "black"):
        parent_dir = filter_color
    
    else:
        parent_dir = str(lower_threshold[0]) + str(upper_threshold[0])
    
    if not os.path.exists(os.path.join(dest_directory, parent_dir, "train", "train")):
        os.makedirs(os.path.join(dest_directory, parent_dir, "train", "train"))
        
    if not os.path.exists(os.path.join(dest_directory, parent_dir, "test", "test")):
        os.makedirs(os.path.join(dest_directory, parent_dir, "test", "test"))
    
    

    

    i = 0
    for image in os.listdir(source_directory):
        
        files = glob.glob(os.path.join(temp_directory,"*"))
        for f in files:
            os.remove(f)
        
        rand = random.uniform(0, 1)
        
        slice_images(
            source_directory,
            image,
            tile_height,
            tile_width)
        
    
        for image in os.listdir(temp_directory):
            
            
            try:
                processed_img = convert_to_binary(os.path.join(
                    temp_directory, image), lower_threshold, upper_threshold, 
                    filter_color)
                
                processed_img = remove_small_regions(processed_img,
                                                     min_region_size)
                processed_img = processed_img*255                    

                if rand <= test_prop:
                    cv2.imwrite(os.path.join(dest_directory, parent_dir, "test", "test",
                                             image),
                                processed_img)
                else:

                    if rotate_tiles:
                        img_rotate_90 = cv2.rotate(processed_img, 
                                                   cv2.ROTATE_90_CLOCKWISE)
                        img_rotate_180 = cv2.rotate(processed_img, 
                                                    cv2.ROTATE_180)
                        img_rotate_270 = cv2.rotate(processed_img,
                                                    cv2.ROTATE_90_COUNTERCLOCKWISE)
                        
                        cv2.imwrite(os.path.join(dest_directory, parent_dir, "train", 
                                                 "train", image),
                                processed_img)                        
                        cv2.imwrite(os.path.join(dest_directory, parent_dir, "train", 
                                                 "train", image+"90.png"),
                                img_rotate_90)
                        cv2.imwrite(os.path.join(dest_directory, parent_dir, "train", 
                                                 "train", image+"180.png"),
                                img_rotate_180)
                        cv2.imwrite(os.path.join(dest_directory, parent_dir, "train", 
                                                 "train", image+"270.png"),
                                img_rotate_270)
                        
                    else:
                        cv2.imwrite(os.path.join(dest_directory, parent_dir, "train", 
                                                 "train", image),
                                processed_img)
                        
            except BaseException:
                continue
        
        i = i + 1
        
        
        
def merge_layers(src):
    """
    takes two existing masks and merges them into a 2 layer image
    writes the new image to disk in a new directory
    only works on blue and black masks for now

    Parameters
    ----------
    src : str
        directory containing blue and black masks 
        must have the structure output by the process_all_images method
        new merged folder will be created here


    Returns
    -------
    None.

    """
    
    for phase in ["train", "test"]:
    
        if not os.path.exists(os.path.join(src, "stacked", phase, phase)):
            os.makedirs(os.path.join(src, "stacked", phase, phase))
            

        for fileName in os.listdir(os.path.join(src, "blue", phase , phase)):
            black_img  = cv2.imread(os.path.join(src, "black", phase, phase, fileName))
            blue_img = cv2.imread(os.path.join(src, "blue", phase, phase, fileName))
                
            black_layer = black_img[:,:,0]
            blue_layer = blue_img[:,:,0]
            blank_layer = black_img[:,:,2]
                
            stacked_img = np.stack((black_layer, blue_layer, blank_layer), axis=2)
            cv2.imwrite(os.path.join(src, "stacked", phase, phase, fileName), stacked_img)
        
        