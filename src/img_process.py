PATH_PROCESSED_IMG = "ProcessedImages/"

import cv2
import numpy as np
import datetime

class ImgProcess():
    def __init__(self):
        pass

    #------------------------------------------------------------------------------------------------------
    def process(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")

        crp_img = self.cutStrip(image)
        single_imgs = self.cutSingleImgs(crp_img)
        #self.saveImg(crp_img)
        exit()

    #------------------------------------------------------------------------------------------------------
    def cutStrip(self, image):
        """Cuts a rectangular strip from the input image based on corner detection.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The cropped image strip.

        Notes:
            The method performs the following steps:
            1. Convert the image to grayscale.
            2. Apply Gaussian Blur to the grayscale image.
            3. Define a kernel for morphological operations.
            4. Apply threshold based on Otsu's method.
            5. Close holes and remove blind spots in the thresholded image.
            6. Detect corners with the goodFeaturesToTrack function.
            7. Calculate the dimensions of the rectangle formed by the detected corners.
            8. Perform perspective transformation to obtain a rectangular cropped image.

        Raises:
            ValueError: If the input image is not provided or is not a valid NumPy array.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray_image, (7, 7), 1)

        # Define Kernel Size
        kernel = np.ones((35, 35), np.uint8)

        # Apply threshold based on Otsu
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        # Close von Holes, delete blind Spots
        closed_image_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        edges_otsu = cv2.Canny(closed_image_otsu, 30, 150)

        # detect corners with the goodFeaturesToTrack function.
        corners = cv2.goodFeaturesToTrack(closed_image_otsu, 4, 0.5, 10)
        corners = np.int0(corners)

        pt_A = corners[0][0]
        pt_B = corners[1][0]
        pt_C = corners[2][0]
        pt_D = corners[3][0]

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        max_width = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        max_height = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                 [0, max_height - 1],
                                 [max_width - 1, max_height - 1],
                                 [max_width - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        croped_image = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LINEAR)

        return croped_image

    #------------------------------------------------------------------------------------------------------
    def cutSingleImgs(self, img):
        """Cut and save individual images separated by vertical lines.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            None

        Raises:
            ValueError: If the input image is not provided or is not a valid NumPy array.
        """
        # Check if the image was successfully loaded
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")
        
        # Create a copy of the input image
        negative = img.copy()

        # Calculate column sums
        column_sums = np.sum(negative, axis=0)

        # Find split points (columns with pixel sum below a threshold)
        split_points = np.where(column_sums > 12500)[0]

        # Draw separation lines in the original image
        for split_point in split_points:
            cv2.line(negative, (split_point, 0), (split_point, negative.shape[0]), (0, 255, 0), 3)  # Draw separation lines (Color: 255)
        
        # Cut and save images between separation lines
        for i in range(len(split_points) - 1):
            start_row = split_points[i]
            end_row = split_points[i + 1]

            if end_row - start_row < 10 or end_row is None:
                pass
            else:
                # Crop the image between separation lines
                cropped_image = img[:, start_row:end_row]

                # Save the cropped image (here as PNG, but you can adjust the format)
                self.saveImg(cropped_image, "_" + str(i))

    #------------------------------------------------------------------------------------------------------
    def invertImg(self, img):
        pass

    #------------------------------------------------------------------------------------------------------
    def saveImg(self, img, filename_tag=""):
        """Save an image with a filename containing the current date and time.

        Args:
            img (numpy.ndarray): The image to be saved.
            filename_tag (str): Tag to be added to the filename.

        Returns:
            None

        Raises:
            IOError: If the image cannot be successfully saved.
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = PATH_PROCESSED_IMG + f"{current_date}_IMG" + filename_tag + ".jpg"

        try:
            cv2.imwrite(filename, img)
        except Exception as e:
            raise IOError(f"Error saving the image to '{filename}': {str(e)}")

    #------------------------------------------------------------------------------------------------------
    def showImg(self, img):
        pass