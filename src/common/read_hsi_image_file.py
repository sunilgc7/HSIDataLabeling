import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..utils.rectangle_drawer import *


start_point = None
end_point = None
drawing = False

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing, image

    # On left mouse button down, record the starting coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    # On mouse move, update the ending coordinates and draw the rectangle
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            temp_image = image.copy()
            cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Image", temp_image)

    # On left mouse button up, record the ending coordinates and draw the final rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Image", image)

def read_hsi(image_dir_path,callback):
    print("*******************************")
    global image
    dir_list = os.listdir(image_dir_path)
    filtered_list = [x for x in dir_list if "png" in x]
    folder = filtered_list[0].split(".")[0]

    dark_ref_hdr_path = image_dir_path+"/capture/DARKREF_"+folder+".hdr"
    dark_ref_raw_path = image_dir_path+"/capture/DARKREF_"+folder+".raw"

    white_ref_hdr_path = image_dir_path+"/capture/WHITEREF_"+folder+".hdr"
    whie_ref_raw_path = image_dir_path+"/capture/WHITEREF_"+folder+".raw"

    data_hdr_path = image_dir_path+"/capture/"+folder+".hdr"
    data_raw_path = image_dir_path+"/capture/"+folder+".raw"

    dark_ref = envi.open(dark_ref_hdr_path, dark_ref_raw_path)
    white_ref = envi.open(white_ref_hdr_path, whie_ref_raw_path)
    data_ref = envi.open(data_hdr_path, data_raw_path)

    white_ref_array = np.array(white_ref.load())
    dark_ref_array = np.array(dark_ref.load())
    data_ref_array = np.array(data_ref.load())

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
    drawer = RectangleDrawer(image,clone)
    drawer.draw_rectangle()

    # After drawing, get the coordinates of the rectangle
    start, end = drawer.get_coordinates()
    print(f"Start Point: {start}, End Point: {end}")
        
