import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt

def read_hsi(image_dir_path):
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

    im_channel = data_ref_array[..., 95]
    plt.style.use('ggplot')
    plt.imshow(im_channel)
    plt.axis(False)
    plt.savefig('band.jpeg', dpi=500)
    plt.show()