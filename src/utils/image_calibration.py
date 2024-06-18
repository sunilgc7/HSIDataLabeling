import os
import argparse
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations


def get_calibrated_image(subdir_lvl1):
        
        dir_list = os.listdir(subdir_lvl1)
        filtered_list = [x for x in dir_list if "png" in x]
        folder = filtered_list[0].split(".")[0]

        dark_ref_hdr_path = subdir_lvl1+"/capture/DARKREF_"+folder+".hdr"
        dark_ref_raw_path = subdir_lvl1+"/capture/DARKREF_"+folder+".raw"

        white_ref_hdr_path = subdir_lvl1+"/capture/WHITEREF_"+folder+".hdr"
        whie_ref_raw_path = subdir_lvl1+"/capture/WHITEREF_"+folder+".raw"

        data_hdr_path = subdir_lvl1+"/capture/"+folder+".hdr"
        data_raw_path = subdir_lvl1+"/capture/"+folder+".raw"

        dark_ref = envi.open(dark_ref_hdr_path, dark_ref_raw_path)
        white_ref = envi.open(white_ref_hdr_path, whie_ref_raw_path)
        data_ref = envi.open(data_hdr_path, data_raw_path)

        white_ref_array = np.array(white_ref.load())
        dark_ref_array = np.array(dark_ref.load())
        data_ref_array = np.array(data_ref.load())

        # mean
        white_avg = np.mean(white_ref_array, axis=0, keepdims=True)
        dark_avg = np.mean(dark_ref_array, axis=0, keepdims=True)

        #calibration
        calibrated_img = (np.divide(np.subtract(data_ref_array, dark_avg),np.subtract(white_avg, dark_avg)))

        return calibrated_img