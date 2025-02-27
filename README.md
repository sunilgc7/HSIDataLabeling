# HSI
This is a repository to develop tools for hyperspectral image processing targeted for FX17.
To run the script use:
Install python envrionment:
1. Clone the project
2. Navigate inside the project directory
3. Create python virtual envrionment
4. Activate python virtual envrionment
5. Install python liabraries using requirements.txt file
6. Run the script as python label_roi.py -p "path to the hyperspectral data folder"  -l <path to label directory>

eg. in window terminal

python .\label_roi.py -p "F:\mydata\HSI\10_G116R4_2024-04-11_19-06-36" -l labeled_data

It will create a windows with calibrated psuedo rgb image. You can create box or roi of your intrest using mouse then you need to type the label name in terminal as intended class name
![Screenshot 2025-02-27 152730](https://github.com/user-attachments/assets/fd7f503e-c1f4-4006-9c94-c9556c5575c1)
