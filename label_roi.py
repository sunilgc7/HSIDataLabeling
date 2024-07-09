from src.common.read_hsi_image_file import *
import argparse

parser = argparse.ArgumentParser(
        description="DL segmentation model training")
parser.add_argument("-p",
                        "--im_path",
                        help="Path for HSI image",
                        type=str)
parser.add_argument("-l",
                        "--lbl_path",
                        help="Label path",
                        type=str)
    
args = parser.parse_args()

image_dir_path = args.im_path
label_image_folder = args.lbl_path

read_hsi(image_dir_path,label_image_folder)

