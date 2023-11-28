import cv2
import numpy as np

# Define a class for image processing
class ImageProcessing():
    def __init__(self):
        # Initialize threshold values
        self.threshold_value_low = 150
        self.threshold_value_high = 255
        self.Image = None

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

        # Apply threshold based on trackbar values
        _, thresh = cv2.threshold(blur, self.threshold_value_low, self.threshold_value_high, cv2.THRESH_BINARY)

        # Define Kernel Size
        kernel = np.ones((15, 15), np.uint8)

        # Close von Holes, delete blind Spots
        closed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(closed_image, 30, 150)

        cv2.imshow('Thresh', closed_image)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            copy = image.copy()
            cropedImage = copy[y:y+h,x:x+w]
            cv2.imshow("test", cropedImage)
            # Draw green rectangles around contours
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Draw only the first contour

        # Prepare text and position for display
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 4
        font_thickness = 2
        text_y = 100

        # Display text on images
        text_original_size = cv2.getTextSize('Original', font, font_size, font_thickness)
        text_edges_size = cv2.getTextSize('Edges', font, font_size, font_thickness)

        text_original_x = int((image.shape[1] - text_original_size[0][0]) / 2)
        text_edges_x = int((edges.shape[1] - text_edges_size[0][0]) / 2)

        cv2.putText(image, 'Original', (text_original_x, text_y), font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(edges, 'Edges', (text_edges_x, text_y), font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Concatenate images for display
        result = np.hstack((image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('Image and Edges: Press ESC to close the Window', result)
