import src.colorcorectionSW as ccSW
import src.colorcorrectionCOLOR as ccC
import cv2


def main():
    #ccSW.bitwise('ProcessedImages/2023-12-19_18-31-24_IMG_0.jpg',showImage=True)

    snippedsw = cv2.imread('ProcessedImages/Snipped.jpg')
    img = cv2.imread('ProcessedImages/2023-12-19_18-31-24_IMG_5.jpg')
    referenz = cv2.imread('Images/Referenz/Analogscan043.jpg')
    offset = ccSW.calcOffset(snippedsw)
    ccSW.invert_with_offset(img, offset, showImage=True)


if __name__ == '__main__':
    main()