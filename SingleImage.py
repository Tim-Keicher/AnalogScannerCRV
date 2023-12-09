from src import image_processing as ip
import numpy as np
import cv2
global mode
imageMode = "image"
cameraMode = "camera"

# Specify the path to the image
image_path = 'Images/35mmSW.jpg'
imageP = ip.ImageProcessing()
scale_percent = 40

def  calcOffset(snipped):
    img = cv2.cvtColor(snipped, cv2.COLOR_BGR2GRAY)

    average_value = int(img.mean())
    print(f"Average Pixelvalue: {average_value}")

    return average_value


def invert_with_offset(img, offset):
    # In Graustufen umwandeln, falls das Bild nicht bereits in Graustufen ist
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invertieren und Offset anwenden
    inverted_img = 255 - img + offset
    inverted_img = np.clip(inverted_img, 0, 255).astype(np.uint8)  # Wertebereich auf 0-255 begrenzen

    # Bild anzeigen (optional)
    cv2.imshow('Inverted with Offset', inverted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return inverted_img



def main():
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Unable to load the image from {}".format(image_path))
        exit()

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    imageP.setImage(image)

    image = imageP.getImage()

    imageP.processImage(image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    snipped = cv2.imread('ProcessedImages/Snipped.png')
    offset = calcOffset(snipped)

    test = cv2.imread('ProcessedImages/Bild_25.png')
    invert_with_offset(test, offset)

if __name__ == '__main__':
    main()
