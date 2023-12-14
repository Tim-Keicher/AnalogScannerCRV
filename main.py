# This is part of the lecture roboter vision
# Date 10.11.2023

from src import img_process as ip
import cv2
global mode
imageMode = "image"
cameraMode = "camera"

# Specify the path to the image
image_path = 'Images/35mmSW.jpg'
imageP = ip.ImgProcess()

scale_percent = 20

def main():
    cap = None

    if mode is cameraMode:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open the camera.")
            # Close if there is no Webcam available
            exit()
    elif mode is imageMode:
        image = cv2.imread(image_path)
        if image is None:
            print("[ERROR] Unable to load the image from {}".format(image_path))
            exit()
        # Ursprüngliche Abmessungen des Bildes erhalten
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        imageP.process(image)
        #imageP.setImage(image)

    else:
        print("[ERROR] Mode {} unknown".format(mode))
        exit()

    # Generate and resize windows
#    cv2.namedWindow('Image and Edges: Press ESC to close the Window', cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Image and Edges: Press ESC to close the Window', 1080, 360)
    # cv2.namedWindow('inverted_image: Press ESC to close the Window', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('inverted_image: Press ESC to close the Window', 1080, 720)

    
    #cv2.namedWindow('Trackbar', cv2.WINDOW_NORMAL)
    #cv2.createTrackbar('Threshold Low', 'Trackbar', imageP.threshold_value_low, 255, imageP.on_trackbar_low)
    #cv2.createTrackbar('Threshold High', 'Trackbar', imageP.threshold_value_high , 255, imageP.on_trackbar_high)
    

    ### Mainloop
    while True:
        if mode is cameraMode:
            ret, image = cap.read()
            if not ret:
                print("[ERROR] Could not read frame.")
                break
        else:
            #image = imageP.getImage()
            pass

        #imageP.processImage(image)

        if cv2.waitKey(1) == 27:
            if mode is cameraMode:
                # Release the VideoCapture object and close the window
                cap.release()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("What Kind of Image do you want to load?")
    print("[1] Image from path")
    print("[2] Image of (connected) camera")
    print("[9] Exit")

    # Check the choosen mode
    #answer = input("Insert number:")
    answer = "1"
    if answer == "1":
        mode = imageMode
        main()
        exit()
    elif answer == "2":
        mode = cameraMode
        main()
        exit()
    elif answer == "9":
        exit()
    else:
        print("Your chosen mode is not available, choose another one")


