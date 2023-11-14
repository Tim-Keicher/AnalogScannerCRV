import cv2
import numpy as np

def processImage(image):
    # Get a grayscaleimage
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect Edges based on Cannyfilter
    edges = cv2.Canny(gray_image, 30, 150)

    # Invert the SW image
    inverted_image = cv2.bitwise_not(image)

    cv2.imshow('inverted_image: Press ESC to close the Window', inverted_image)

    # Insert Text to Images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 4
    font_thickness = 2
    text_y = 100

    # Insert description into middle
    text_original_size = cv2.getTextSize('Original', font, font_size, font_thickness)
    text_edges_size = cv2.getTextSize('Eges', font, font_size, font_thickness)

    text_original_x = int((image.shape[1] - text_original_size[0][0]) / 2)
    text_edges_x = int((edges.shape[1] - text_edges_size[0][0]) / 2)

    cv2.putText(image, 'Original', (text_original_x, text_y), font, font_size, (255, 255, 255), font_thickness,
                cv2.LINE_AA)
    cv2.putText(edges, 'Eges', (text_edges_x, text_y), font, font_size, (255, 255, 255), font_thickness,
                cv2.LINE_AA)

    # Display both images in one window
    result = np.hstack((image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

    # Display the result in a single window
    cv2.imshow('Image and Edges: Press ESC to close the Window', result)