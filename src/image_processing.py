import cv2
import numpy as np

# Define a class for image processing
class ImageProcessing():
    def __init__(self):
        # Initialize threshold values
        self.threshold_value_low = 0#150
        self.threshold_value_high = 255
        self.Image = None
        self.cropedImage = None

    def getImage(self):
        return self.Image

    def setImage(self,img):
        self.Image = img

    # Callback functions for trackbar changes
    def on_trackbar_low(self, val):
        self.threshold_value_low = cv2.getTrackbarPos('Threshold Low', 'Trackbar')

    def on_trackbar_high(self, val):
        self.threshold_value_high = cv2.getTrackbarPos('Threshold High', 'Trackbar')

    # Process the image
    def processImage(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray_image, (7, 7), 1)

        # Define Kernel Size
        kernel = np.ones((15, 15), np.uint8)

        # Apply threshold based on Otsu
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        # Close von Holes, delete blind Spots
        closed_image_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        edges_otsu = cv2.Canny(closed_image_otsu, 30, 150)
        contours_otsu, _ = cv2.findContours(edges_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_otsu:
            x, y, w, h = cv2.boundingRect(contour)
            self.cropedImage = image[y:y+h,x:x+w]
            if w < h:
                self.cropedImage = cv2.rotate(self.cropedImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            #cv2.imwrite("Images/35mmSW_cropped.jpg", self.cropedImage)
#            cv2.imshow("ROI_otsu",  self.cropedImage)
            # Draw green rectangles around contours
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Draw only the first contour

        '''
        # Apply threshold based on trackbar values
        _, thresh_binary = cv2.threshold(blur, self.threshold_value_low, self.threshold_value_high, cv2.THRESH_BINARY)
        # Close von Holes, delete blind Spots
        closed_image = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)

        edges_binary = cv2.Canny(closed_image, 30, 150)
        contours_binary, _ = cv2.findContours(edges_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        cv2.imshow('Thresh', closed_image)

        # Find contours
        for contour in contours_binary:
            x, y, w, h = cv2.boundingRect(contour)
            self.cropedImage = image[y:y+h,x:x+w]
            if w < h:
                self.cropedImage = cv2.rotate(self.cropedImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow("ROI",  self.cropedImage)
            # Draw green rectangles around contours
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Draw only the first contour
        '''

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray_image, (7, 7), 1)

        # Define Kernel Size
        kernel = np.ones((15, 15), np.uint8)

        # Apply threshold based on Otsu
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        # Close von Holes, delete blind Spots
        closed_image_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        edges_otsu = cv2.Canny(closed_image_otsu, 30, 150)
        contours_otsu, _ = cv2.findContours(edges_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # start vertical lines
        croped = self.cropedImage
        croped_gray_image = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)
        croped_blur = cv2.GaussianBlur(croped_gray_image, (7, 7), 1)

        kernel = np.ones((15, 15), np.uint8)

        _, thresh_otsu_croped = cv2.threshold(croped_blur, self.threshold_value_low, self.threshold_value_high, cv2.THRESH_OTSU)
        closed_image_otsu_cropped = cv2.morphologyEx(thresh_otsu_croped, cv2.MORPH_CLOSE, kernel)
        croped_edges = cv2.Canny(closed_image_otsu_cropped, 100, 150, apertureSize=3)
        #contours_otsu_cropped, _ = cv2.findContours(croped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow("croped_edges",croped_edges)
        '''lines = cv2.HoughLinesP(croped_edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(croped, (x1, y1), (x2, y2), (0, 255, 0), 2)'''

        # Zeige das Bild mit den erkannten Linien
        cv2.imshow("Vertical Lines", croped)



        # Prepare text and position for display
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 4
        font_thickness = 2
        text_y = 100

        # Display text on images
        # text_original_size = cv2.getTextSize('Original', font, font_size, font_thickness)
        # text_edges_size = cv2.getTextSize('Edges', font, font_size, font_thickness)

        # text_original_x = int((image.shape[1] - text_original_size[0][0]) / 2)
        # text_edges_x = int((edges.shape[1] - text_edges_size[0][0]) / 2)

        # cv2.putText(image, 'Original', (text_original_x, text_y), font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
        # cv2.putText(edges, 'Edges', (text_edges_x, text_y), font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Concatenate images for display
        result = np.hstack((image, cv2.cvtColor(edges_otsu, cv2.COLOR_GRAY2BGR)))
        #cv2.imshow('Image and Edges: Press ESC to close the Window', result)
