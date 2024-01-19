import cv2
import numpy as np
import datetime
import src.colorcorectionBW as ccBW
import src.colorcorrectionCOLOR as ccC

import src.namespace as names

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
        self.PATH_PROCESSED_IMG = "Saves/"
        self.config_cut_img_sep_lines = True

        self.ns = names.Names()
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
    def cutStrip(self, img, boundaryType, visualizeSteps=False):
        """Cuts a rectangular strip from the input image based on corner detection.

        Args:
            img (numpy.ndarray): The input image.
            boundaryType (string): Defines what kind of film size ['DIAS', '35', '120']
            visualizeSteps(boolean): Activate the visualisation of all Steps

        Returns:
            final_array[numpy.ndarray] One or more cropped image strips.

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
        output = img.copy()
        croped_img_array = []

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray_img, (7, 7), 1)

        # Apply threshold based on Otsu
        tr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        _, thresh_otsu = cv2.threshold(blur, tr + 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(cv2.bitwise_not(thresh_otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f'[DEBUG] {len(contours)} contours found')

        # Generate a mask to define where the objects are
        mask = np.zeros_like(img)
        mask = cv2.bitwise_not(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        filtered_contours = []
        for cont in contours:
            if boundaryType == self.ns.name_dia:
                min_size = self.ns.min_size_dia
            else:
                min_size = self.ns.min_size_film

            if cont.size >= min_size:
                cv2.drawContours(mask, [cont], -1, (0, 0, 0), thickness=cv2.FILLED)
                filtered_contours.append(cont)
            else:
                # Speed up the process so we dont loop over
                break
        print(f'[DEBUG] {len(filtered_contours)} Filtered Contours found')

        for cont in filtered_contours:
            # Check for Dias
            if boundaryType == self.ns.name_dia:
                rect = cv2.minAreaRect(cont)
                corners = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(output, [corners], 0, (0, 255, 0), 2)

            else:
                epsilon = 0.02 * cv2.arcLength(cont, True)
                corners_approx = cv2.approxPolyDP(cont, epsilon, True)
                n_corners = len(corners_approx)
                if n_corners == 4:
                    corners = [tuple(point[0]) for point in corners_approx]

            for point in corners:
                x, y = point
                cv2.circle(output, (x, y), 15, (0, 0, 255), -1)

            sorted_coordinates = sorted(corners, key=lambda coord: coord[0])
            left_coordinates = sorted_coordinates[:2]
            right_coordinates = sorted_coordinates[2:]

            # top left corner
            pt_A = sorted(left_coordinates, key=lambda coord: coord[1])[0]
            # bottom left corner
            pt_B = sorted(left_coordinates, key=lambda coord: coord[1])[1]
            # bottom right corner
            pt_C = sorted(right_coordinates, key=lambda coord: coord[1])[1]
            # top right corner
            pt_D = sorted(right_coordinates, key=lambda coord: coord[1])[0]

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
            warped_perspective = cv2.warpPerspective(img, M, (max_width, max_height), flags=cv2.INTER_LINEAR)
            croped_img_array.append(warped_perspective)

        if visualizeSteps:
            self.showImg('thresh_otsu', thresh_otsu)
            self.showImg('mask', mask)
            self.showImg('output', output)

        final_array = self.resizeStripByType(img_array=croped_img_array, boundaryType=boundaryType,
                                             visualizeSteps=visualizeSteps)

        return final_array

    def resizeStripByType(self, img_array, boundaryType, visualizeSteps=False):
        """Resize the strips to cut off the edge

                Args:
                    img_array[numpy.ndarray]: The array of inputs.
                    boundaryType (string): Defines what kind of film size ['DIAS', '35', '120']
                    visualizeSteps(boolean): Activate the visualisation of all Steps

                Returns:
                    cutStrip[numpy.ndarray]: One or more cropped image strips.

                Notes:
                """
        cutStrip = []
        for img in img_array:
            h, w = img.shape[:2]
            if boundaryType == self.ns.name_small_format:
                cuttingHeight = int(h * 15 / 100)
                cuttingWidth = int(cuttingHeight * 50 / 100)
                cutStrip.append(img[cuttingHeight:h - cuttingHeight, cuttingWidth:w - cuttingWidth])
            elif boundaryType == self.ns.name_medium_format:
                cuttingHeight = int(h * 4 / 100)
                cuttingWidth = int(cuttingHeight * 90 / 100)
                cutStrip.append(img[cuttingHeight:h - cuttingHeight, cuttingWidth:w - cuttingWidth])
            elif boundaryType == self.ns.name_dia:
                cuttingHeight = int(h*5/100)
                cuttingWidth = int(w*5/100)
                precut_img = img[cuttingHeight:h - cuttingHeight, cuttingWidth:w - cuttingWidth]
                gray_img = cv2.cvtColor(precut_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray_img, (7, 7), 1)
                _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

                h, w = precut_img.shape[:2]
                horizontal_mask, vertical_mask, horizontal_corners, vertical_corners = self.generate_dia_masks(w=w, h=h, visualizeSteps=visualizeSteps)
                alignment = self.get_dia_alignment(threshold_img=thresh_otsu, horizontal_mask=horizontal_mask, vertical_mask=vertical_mask, visualizeSteps=visualizeSteps)

                if alignment == self.ns.alignment_vertical:
                    dia = precut_img[vertical_corners[0][1]:vertical_corners[2][1], vertical_corners[0][0]:vertical_corners[2][0]]
                    # Rotate 90 degree
                    dia = cv2.rotate(dia, cv2.ROTATE_90_CLOCKWISE)
                else:
                    dia = precut_img[horizontal_corners[0][1]:horizontal_corners[2][1],
                          horizontal_corners[0][0]:horizontal_corners[2][0]]
                cutStrip.append(dia)
            else:
                print(f'[WARNING] Unknown BoundaryType: {boundaryType}')
        return cutStrip

    def generate_dia_masks(self, w, h, visualizeSteps=False):
        # Initialize horizontal and vertical masks
        horizontal_mask = np.zeros((h, w), dtype=np.uint8)
        vertical_mask = np.zeros((h, w), dtype=np.uint8)

        # Define dimensions for horizontal and vertical masks
        horizontal_mask_width = int(w * 80 / 100)
        horizontal_mask_height = int(w * 50 / 100)
        vertical_mask_width = int(w * 50 / 100)
        vertical_mask_height = int(w * 80 / 100)

        # Calculate starting coordinates for both masks
        horizontal_mask_x = (w - horizontal_mask_width) // 2
        horizontal_mask_y = (h - horizontal_mask_height) // 2
        vertical_mask_x = (w - vertical_mask_width) // 2
        vertical_mask_y = (h - vertical_mask_height) // 2

        # Draw rectangles on the masks
        cv2.rectangle(horizontal_mask, (horizontal_mask_x, horizontal_mask_y),
                      (horizontal_mask_x + horizontal_mask_width, horizontal_mask_y + horizontal_mask_height), 255, -1)
        cv2.rectangle(vertical_mask, (vertical_mask_x, vertical_mask_y),
                      (vertical_mask_x + vertical_mask_width, vertical_mask_y + vertical_mask_height), 255, -1)

        # Define corner points for both masks
        horizontal_corners = [(horizontal_mask_x, horizontal_mask_y),
                              (horizontal_mask_x + horizontal_mask_width, horizontal_mask_y),
                              (horizontal_mask_x + horizontal_mask_width, horizontal_mask_y + horizontal_mask_height),
                              (horizontal_mask_x, horizontal_mask_y + horizontal_mask_height)]

        vertical_corners = [(vertical_mask_x, vertical_mask_y),
                            (vertical_mask_x + vertical_mask_width, vertical_mask_y),
                            (vertical_mask_x + vertical_mask_width, vertical_mask_y + vertical_mask_height),
                            (vertical_mask_x, vertical_mask_y + vertical_mask_height)]
        if visualizeSteps:
            self.showImg("horizontal mask", horizontal_mask)
            self.showImg("vertical mask", vertical_mask)
        # Return masks and corner points
        return horizontal_mask, vertical_mask, horizontal_corners, vertical_corners

    def get_dia_alignment(self, threshold_img, vertical_mask, horizontal_mask, visualizeSteps=False):

        hor_masked_image = cv2.bitwise_and(threshold_img, threshold_img, mask=horizontal_mask)
        hor_matching_pixels = np.count_nonzero(hor_masked_image)

        vert_masked_image = cv2.bitwise_and(threshold_img, threshold_img, mask=vertical_mask)
        vert_matching_pixels = np.count_nonzero(vert_masked_image)

        if hor_matching_pixels > vert_matching_pixels:
            alignment = self.ns.alignment_horizontal
        else:
            alignment = self.ns.alignment_vertical

        if visualizeSteps:
            self.showImg('vert_masked_image', vert_masked_image)
            self.showImg('hor_masked_image', hor_masked_image)
            print(f'[INFO] The Dia got detected as {alignment}.')

        return alignment

    # ------------------------------------------------------------------------------------------------------
    def cutSingleImgs(self, img, boundaryType, visualizeSteps=False):
        """Cut and save individual images separated by vertical lines.

        Args:
            img (numpy.ndarray): The input image.
            boundaryType (string): Defines what kind of film size ['DIAS', '35', '120']
            visualizeSteps(boolean): Activate the visualisation of all Steps

        Returns:
            cropped_imgs[numpy.ndarray]: A list of cropped images between separation lines.
            strip(numpy.ndarray): Cut strip to calculate the offset

        Raises:
            ValueError: If the input image is not provided or is not a valid NumPy array.
        """
        # Check if the image was successfully loaded
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid input image. Please provide a valid NumPy array.")
        output = img.copy()
        h, w = img.shape[:2]
        print(f'[INFO] Film type: {boundaryType}')
        if boundaryType == self.ns.name_small_format:
            ratio = 25 / h
            width_ratio = 36 / ratio
            n_images = int(w / width_ratio)
            singleImageWidth = h * 36 / 25
        elif boundaryType == self.ns.name_dia:
            ratio = 25 / h
            width_ratio = 36 / ratio
            n_images = int(w / width_ratio)
            singleImageWidth = h * 36 / 25
        else:
            ratio = 60 / h
            width_ratio = 90 / ratio
            n_images = int(w / width_ratio)
            singleImageWidth = h * 90 / 60

        print(f'[INFO] Calculated images: {n_images}')
        print(f'[INFO] Calculated single image width: {singleImageWidth}')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # enhanced_edges = cv2.addWeighted(img, 1.5, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
        blur = cv2.GaussianBlur(gray, (7, 7), 1)
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        blacked = thresh_otsu.copy()
        check_black = np.any(thresh_otsu == 0, axis=0)

        # Generate the Blacked mask to detect the images.
        blacked[:, check_black] = 0

        coordinates = []
        for x in range(w - 1):
            # Get the points of the Binary change
            if blacked[0, x] == 0 and blacked[0, x + 1] == 255 or blacked[0, x] == 255 and blacked[0, x + 1] == 0:
                coordinates.append(x)

        # Filter lines generate them on output
        filtered_coords = []
        for coord in coordinates:
            if coord <= 10 or coord >= singleImageWidth * 90 / 100:
                filtered_coords.append(coord)
                cv2.line(output, (coord, 0), (coord, output.shape[0]), (0, 255, 0), 3)

        img_masks = []

        cropped_imgs = []
        strip = None
        firstImg = False

        for i in range(0, len(filtered_coords) - 1):
            if filtered_coords[i] <= singleImageWidth * 85 / 100:
                pass
            else:
                if filtered_coords[i] >= singleImageWidth * 85 / 100 and firstImg is False:
                    cropped_imgs.append(img[:, 0:filtered_coords[i]])
                    firstImg = True
                elif i == len(filtered_coords) - 2 and w -filtered_coords[i + 1] >= singleImageWidth * 85 / 100:
                    cropped_imgs.append(img[:, filtered_coords[i + 1]:w])

                ofs = filtered_coords[i + 1] - filtered_coords[i]

                if ofs > singleImageWidth * 85 / 100:
                    cropped_imgs.append(img[:, filtered_coords[i]:filtered_coords[i + 1]])
                else:
                    if strip is None:
                        strip = img[:, filtered_coords[i]:filtered_coords[i + 1]]

                if w - filtered_coords[i + 1] <= singleImageWidth * 85 / 100:
                    break

        if visualizeSteps:
            self.showImg("Detect Single Images", img, destroy_window=False)
            stacked = np.concatenate((img, cv2.cvtColor(thresh_otsu, cv2.COLOR_GRAY2BGR)), axis=0)
            self.showImg("Detect Single Images", stacked, destroy_window=False)
            stacked = np.concatenate((stacked, cv2.cvtColor(blacked, cv2.COLOR_GRAY2BGR)), axis=0)
            self.showImg("Detect Single Images", stacked, destroy_window=False)
            stacked = np.concatenate((stacked, output), axis=0)
            self.showImg("Detect Single Images", stacked, destroy_window=False)

            for coordinate in filtered_coords:
                img_masks.append(
                    self.generate_img_masks(img_height=int(h), img_length=int(singleImageWidth), strip_length=int(w),
                                            pos=coordinate, visualizeSteps=visualizeSteps))
            _ = self.val_img_positions(single_img_masks=img_masks, strip_mask=blacked, pos=filtered_coords,
                                       img_length=singleImageWidth, visualizeSteps=visualizeSteps)
            cv2.destroyAllWindows()

        return cropped_imgs, strip

    def generate_img_masks(self, img_height, img_length, strip_length, pos, visualizeSteps=False):
        img_mask = np.zeros((img_height, strip_length), dtype=np.uint8)

        # Define dimensions for horizontal and vertical masks
        img_mask_width = img_length
        img_mask_height = img_height

        img_mask_x = pos - img_length
        img_mask_y = 0

        if pos + img_length < strip_length:
            # Draw rectangles on the masks
            cv2.rectangle(img_mask, (img_mask_x, img_mask_y),
                          (img_mask_x + img_mask_width, img_mask_y + img_mask_height), 255, -1)

        img_mask = cv2.bitwise_not(img_mask)
        if visualizeSteps:
            #self.showImg('image_mask', img_mask, destroy_window=False)
            pass

        return img_mask

    def val_img_positions(self, single_img_masks, strip_mask, pos, img_length, visualizeSteps=False):
        mask_positions = []
        for single_mask in single_img_masks:
            result = cv2.bitwise_or(strip_mask, single_mask)
            result_xor = cv2.bitwise_not(cv2.bitwise_xor(result, single_mask))
            mask_positions.append(np.count_nonzero(result_xor))

            # Generate output
            if visualizeSteps:
                stacked = np.concatenate((strip_mask, single_mask, result, result_xor), axis=0)
                self.showImg('stacked', stacked, destroy_window=False)

        # img witch should be good
        golden_img_idx = np.argmax(mask_positions)
        val = pos[golden_img_idx]
        print(val)
        return val

    # ------------------------------------------------------------------------------------------------------
    def invertImg(self, negative_img, offset_img, negative_type, visualizeSteps=False):
        """ Invert the provided image and return it, based on the negative_type.

                Args:
                    negative_img (numpy.ndarray): The negative input image.
                    offset_img (numpy.ndarray): Cutout of the negative strip (total white value).
                    negative_type (string): CHeck for Color or BW color detection.
                    visualizeSteps(boolean): Activate the visualisation of all Steps

                Returns:
                    inverted_image (numpy.ndarray): The inverted image

                Raises:
                    ReturnError: If the negative_type is not valid type.
                """
        # Invert colored image
        if negative_type is self.ns.name_negative_color:
            offset = ccC.calcOffset(offset_img, verbose=False)
            inverted_image = ccC.invert_with_offset(img=negative_img, offset=offset, showImage=visualizeSteps)

        # Invert black and white image
        elif negative_type is self.ns.name_negative_bw:
            offset = ccBW.calcOffset(offset_img, verbose=False)
            inverted_image = ccBW.invert_with_offset(img=negative_img, offset=offset, showImage=visualizeSteps)

        else:
            print(f'[WARNING] Type {negative_type} of inverting unknown ')
            inverted_image = -1

        return inverted_image

    # ------------------------------------------------------------------------------------------------------
    def saveImg(self, img, file_path_and_name):
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
        filename = file_path_and_name + "-" + current_date + ".jpg"

        try:
            img = np.array(img)
            cv2.imwrite(filename, img)
        except Exception as e:
            raise IOError(f"Error saving the image to '{filename}': {str(e)}")

    # ------------------------------------------------------------------------------------------------------
    def showImg(self, window_name, img, destroy_window=True):
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        if destroy_window:
            cv2.destroyAllWindows()
