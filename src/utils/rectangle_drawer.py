import cv2
import os

class RectangleDrawer:
    def __init__(self,image,clone,hcube,label_dir):
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.image = image
        self.clone = clone
        self.hcube = hcube
        self.label_dir = label_dir

    def create_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def mouse_callback(self, event, x, y, flags, param):
        i=0
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                temp_image = self.image.copy()
                cv2.rectangle(temp_image, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.imshow("Image", temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False
            cv2.rectangle(self.image, self.start_point, self.end_point, (0, 255, 0), 2)
            print("start ",self.start_point)
            print("end ",self.end_point)
            roi_name = input("Enter name for this ROI: ")
            print(roi_name)
            save_dir_path = self.label_dir+"/"+roi_name
            self.create_directory(save_dir_path)
            cropped_image = self.image[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0]]
            cv2.imwrite(save_dir_path+"/"+roi_name+str(i)+".jpg", cropped_image)
            i=i+1
            cv2.imshow("Image", self.image)

    def draw_rectangle(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        while True:
            cv2.imshow("Image", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.image = self.clone.copy()

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_coordinates(self):
        return self.start_point, self.end_point
