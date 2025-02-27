import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..utils.rectangle_drawer import *
from plantcv import plantcv as pcv

def read_hsi(image_dir_path,label_dir):
    print("*******************************")
    global image
    dir_list = os.listdir(image_dir_path)
    filtered_list = [x for x in dir_list if "png" in x]
    folder = filtered_list[0].split(".")[0]

    dark_ref_raw_path = image_dir_path+"/capture/DARKREF_"+folder+".raw"

    white_ref_raw_path = image_dir_path+"/capture/WHITEREF_"+folder+".raw"

    data_raw_path = image_dir_path+"/capture/"+folder+".raw"

    raw_img        = pcv.readimage(data_raw_path, mode='envi')
    white_ref      = pcv.readimage(white_ref_raw_path, mode='envi')
    dark_ref       = pcv.readimage(dark_ref_raw_path, mode='envi')
    
    calibrateddata = pcv.hyperspectral.calibrate(raw_data = raw_img, white_reference= white_ref, dark_reference= dark_ref)
    data_ref_array = calibrateddata.array_data

    blue_band = data_ref_array[..., 24]
    green_band = data_ref_array[..., 70]
    red_band = data_ref_array[..., 95]


    blue_band_normalized = cv2.normalize(blue_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    green_band_normalized = cv2.normalize(green_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    red_band_normalized = cv2.normalize(red_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    print(blue_band)

    image = cv2.merge((blue_band_normalized, green_band_normalized, red_band_normalized))
    clone = image.copy()

    # Usage
    drawer = RectangleDrawer(image,clone,data_ref_array,label_dir)
    drawer.draw_rectangle()

    # After drawing, get the coordinates of the rectangle
    start, end = drawer.get_coordinates()
    print(f"Start Point: {start}, End Point: {end}")
        
