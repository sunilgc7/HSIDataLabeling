import pytest
from scipy import stats
from ..src.common.read_hsi_image_file import *

image_dir_path = "/home/developer/SGCPostDOC/data/sample/1_G121R3_2024-04-02_18-14-49"

# pytest test_example.py

@pytest.mark.parametrize("image_dir_path", [image_dir_path])
def test_get_hsi_image(image_dir_path):
  """
  This test compares the means of two treatment groups using a two-tailed t-test.
  """
  read_hsi(image_dir_path,mouse_callback)