import pytest
from scipy import stats
from ..src.common.read_hsi_image_file import *

image_dir_path = "/home/pagsun/SunilGCData/2_normal_system/camelina/after acclamatization/HSI/10_G116R4_2024-03-25_17-58-58"

# pytest test_example.py

@pytest.mark.parametrize("image_dir_path", [image_dir_path])
def test_get_hsi_image(image_dir_path):
  """
  This test compares the means of two treatment groups using a two-tailed t-test.
  """
  label_image_folder = "/home/pagsun/SunilGCData/camelina_label_data"
  read_hsi(image_dir_path,label_image_folder)