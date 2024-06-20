import pytest
from scipy import stats
from common.read_hsi_image_file import *

# Sample data (replace with your actual data)
treatment_1_data = [10, 12, 8, 9, 11]
treatment_2_data = [14, 13, 15, 16, 12]

image_dir_path = "/home/developer/SGCPostDOC/data/sample/B3_TR3_R1T11_A3T2_2023-11-30_17-28-37"

# pytest test_example.py


@pytest.mark.parametrize("treatment_1", [treatment_1_data])
@pytest.mark.parametrize("treatment_2", [treatment_2_data])
def test_treatment_means(treatment_1, treatment_2):
  """
  This test compares the means of two treatment groups using a two-tailed t-test.
  """
  # Calculate means for each treatment group
  treatment_1_mean = sum(treatment_1) / len(treatment_1)
  treatment_2_mean = sum(treatment_2) / len(treatment_2)

  # Perform the two-tailed t-test
  t_statistic, p_value = stats.ttest_ind(treatment_1, treatment_2)

  # Assert the p-value is less than a significance level (e.g., 0.05)
  assert p_value < 0.05, f"Treatment means are not statistically different (p-value: {p_value})"


@pytest.mark.parametrize("image_dir_path", [image_dir_path])
def test_get_hsi_image(image_dir_path):
  """
  This test compares the means of two treatment groups using a two-tailed t-test.
  """
  read_hsi(image_dir_path)