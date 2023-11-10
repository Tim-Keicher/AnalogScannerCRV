# This is part of the lecture roboter vision
# Date 10.11.2023

import image_processing as ip
import cv2

# Specify the path to the image
image_path = 'Images/Image1.jpg'

def main_Imageimport():
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image loaded successfully
    if image is None:
        print("[ERROR] Unable to load the image from {image_path}".format(image_path))
    else:
        print("[INFO] Image loaded from {}".format(image_path))

        # Display the image in a window
        cv2.imshow('Press ESC to close the Window', image)

        # Wait for a key event
        key = cv2.waitKey(0)

        # Check if the pressed key is ESC (27: ASCII for ESC)
        if key == 27:
            # Close the window if ESC is pressed
            cv2.destroyAllWindows()

def main_Webcam():
    # Create a VideoCapture object to capture video from the default camera (usually 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open the camera.")
        # Close if there is no Webcam available
        exit()

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            print("[ERROR] Could not read frame.")
            break

        # Display the frame in a window
        cv2.imshow('Webcam, press ESC to close', frame)

        # Break the loop if 'ESC' is pressed (27: ASCII for ESC)
        if cv2.waitKey(1) == 27:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("What Kind of Image do you want to load?")
    print("[1] Image from path")
    print("[2] Image of (connected) camera")
    print("[9] Exit")

    while True:
        # Check the choosen mode
        answer = input("Insert number:")

        if answer == "1":
            main_Imageimport()
            exit()
        elif answer == "2":
            main_Webcam()
            exit()
        elif answer == "9":
            exit()
        else:
            print("Your choosen mode is not available, choose another one")


