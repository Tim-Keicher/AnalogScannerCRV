import cv2
import numpy as np
import datetime
import src.colorcorectionBW as ccBW
import src.colorcorrectionCOLOR as ccC


class ImageProcessing:
    """A class for image processing operations, including strip cutting, individual image extraction, and saving.

    Attributes:
        PATH_PROCESSED_IMG (str): Path for saving processed images.

    Methods:
        process(img)
        cutStrip(img)
        cutSingleImgs(img)
        invertImg(img)
        saveImg(img, filename_tag="")
        showImg(img)
    """

    def __init__(self):
        self.scan_type = None
        self.type_color = 'color'
        self.type_bw = 'bw'

        self.showImage = False
        self.PATH_PROCESSED_IMG = "ProcessedImages/"
        self.config_cut_img_sep_lines = True

    # ------------------------------------------------------------------------------------------------------
    def process(self, img):
        """Process an input image by cutting strips, extracting individual images, and saving them.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            None

        Raises:
            ValueError: If the input image is not provided or is not a valid NumPy array.
        """
        # Check if the image was successfully loaded
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")

        # Cut a strip from the image based on corner detection
        crp_img = self.cutStrip(img)

        # Cut and retrieve individual images separated by vertical lines
        single_imgs, strip = self.cutSingleImgs(crp_img)

        # Save each individual image with an index as part of the filename
        for i, img in enumerate(single_imgs):
            self.saveImg(img, "_" + str(i))

        # Exit the process after saving images
        exit()

    # ------------------------------------------------------------------------------------------------------
    def cutStrip(self, img):
        """Cuts a rectangular strip from the input image based on corner detection.

        Args:
            img (numpy.ndarray): The input image.

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
        # Check if the image was successfully loaded
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray_img, (7, 7), 1)

        # Define Kernel Size
        kernel = np.ones((35, 35), np.uint8)

        # Apply threshold based on Otsu
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        # Close von Holes, delete blind Spots
        closed_img_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        edges_otsu = cv2.Canny(closed_img_otsu, 30, 150)

        # detect corners with the goodFeaturesToTrack function.
        corners = cv2.goodFeaturesToTrack(closed_img_otsu, 4, 0.5, 10)
        corners = np.int0(corners)

        pt_A = corners[0][0]
        pt_B = corners[1][0]
        pt_C = corners[2][0]
        pt_D = corners[3][0]

        # Correct the alignment of the whole stripe
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

        croped_img = cv2.warpPerspective(img, M, (max_width, max_height), flags=cv2.INTER_LINEAR)

        return croped_img

    # ------------------------------------------------------------------------------------------------------
    def cutSingleImgs(self, img):
        """Cut and save individual images separated by vertical lines.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            List[numpy.ndarray]: A list of cropped images between separation lines.

        Raises:
            ValueError: If the input image is not provided or is not a valid NumPy array.
        """
        # Check if the image was successfully loaded
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")

        strip = None

        # Create a copy of the input image
        negative = img.copy()

        # Calculate column sums
        column_sums = np.sum(negative, axis=0)

        # Find split points (columns with pixel sum below a threshold)
        split_points = np.where(column_sums > 25000)[0]

        # Draw separation lines in the original image
        for split_point in split_points:
            cv2.line(negative, (split_point, 0), (split_point, negative.shape[0]), (0, 255, 0), 3)  # Draw separation lines (Color: 255)

        if self.config_cut_img_sep_lines is True:
            self.showImg('cut images: negative with separation lines', negative)

        # Cut and save images between separation lines
        cropped_imgs = []
        for i in range(len(split_points) - 1):
            start_row = split_points[i]
            end_row = split_points[i + 1]

            if end_row - start_row < 10 or end_row is None:
                # Cut a strip to calculate the white balance value of the individual negative film
                if strip is None:
                    strip = img[:, start_row:start_row+9]
                else:
                    pass
            else:
                # Crop the image between separation lines
                cropped_imgs.append(img[:, start_row:end_row])

        return cropped_imgs, strip

    # ------------------------------------------------------------------------------------------------------
    def invertImg(self, negative_img, offset_img, negative_type):
        """ Invert the provided image and return it, based on the negative_type.

                Args:
                    negative_img (numpy.ndarray): The negative input image.
                    offset_img (numpy.ndarray): Cutout of the negative strip (total white value).
                    negative_type (string): CHeck for Color or BW color detection.

                Returns:
                    inverted_image (numpy.ndarray): The inverted image

                Raises:
                    ReturnError: If the negative_type is not valid type.
                """
        # Invert colored image
        if negative_type is self.type_color:
            offset = ccC.calcOffset(offset_img)
            inverted_image = ccC.invert_with_offset(img=negative_img, offset=offset, showImage=self.showImage)

        # Invert black and white image
        elif negative_type is self.type_bw:
            offset = ccBW.calcOffset(offset_img)
            inverted_image = ccBW.invert_with_offset(img=negative_img, offset=offset, showImage=self.showImage)

        else:
            print(f'[WARNING] Type {negative_type} of inverting unknown ')
            inverted_image = -1

        return inverted_image

    # ------------------------------------------------------------------------------------------------------
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
        filename = self.PATH_PROCESSED_IMG + f"{current_date}_IMG" + filename_tag + ".jpg"

        try:
            cv2.imwrite(filename, img)
        except Exception as e:
            raise IOError(f"Error saving the image to '{filename}': {str(e)}")

    #------------------------------------------------------------------------------------------------------
    def showImg(self, window_name, img):
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
