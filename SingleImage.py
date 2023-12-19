from src import image_processing as ip
import numpy as np
import cv2
global mode
import matplotlib.pyplot as plt
from PIL import Image


def  calcOffset(snipped):
    img = cv2.cvtColor(snipped, cv2.COLOR_BGR2GRAY)

    average_value = int(img.mean())
    print(f"Average Pixelvalue: {average_value}")

    return average_value


def invert_with_offset(img, offset):
    # In Graustufen umwandeln, falls das Bild nicht bereits in Graustufen ist
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    print(img.shape)
    # Weißabgleich durchführen
    white_balance_img = cv2.add(img, offset)

    # Inversion der Farben
    inverted_white_balance_img = cv2.bitwise_not(white_balance_img)
    MaxValue = np.max(inverted_white_balance_img)
    factor = 255/MaxValue

    inverted_white_balance_img = np.multiply(inverted_white_balance_img, factor).astype('uint8')
    print(inverted_white_balance_img)
    print(np.max(inverted_white_balance_img))
    data = inverted_white_balance_img

    # Bild mit OpenCV anzeigen
    cv2.imshow('Inverted with Offset', data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return inverted_white_balance_img

def inver_with_offset_color(img):
    # Inversion der Farben für jedes Farbkanalbild
    inverted_img = cv2.bitwise_not(img)

    # Berechnung des wahren Weißwerts für jeden Farbkanal
    true_white = np.max(inverted_img)

    # Offset berechnen
    offset = 255 - true_white

    # Weißabgleich für das gesamte Bild durchführen
    white_balance_img = cv2.add(img, np.array([offset, offset, offset], dtype=np.uint8))

    # Inversion der Farben für das weißabgeglichene Bild
    inverted_white_balance_img = cv2.bitwise_not(white_balance_img)

    # Offset für die invertierten Farben anpassen und lineare Verteilung durchführen
    offset_array = np.ones_like(inverted_img, dtype=np.float32) * offset
    inverted_with_offset = cv2.add(inverted_img.astype(np.float32), offset_array)
    inverted_with_offset = (inverted_with_offset * (255.0 / np.max(inverted_with_offset))).astype(np.uint8)

    # Mittelwert der invertierten und weißabgeglichenen Bilder berechnen
    inverted_combined = cv2.addWeighted(inverted_with_offset, 0.5, inverted_white_balance_img, 0.5, 0)

    # Bild mit OpenCV anzeigen
    cv2.imshow('Inverted with Offset', inverted_white_balance_img)
    cv2.imshow('Inverted with Contrast', inverted_with_offset)
    cv2.imshow('Inverted Combined', inverted_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return inverted_combined

def compare_images(image, original_image):
    # Konvertiere die Bilder in numpy arrays für den Pixelvergleich
    image_array = np.array(image)

    # Bestimme die Größe des Bildes
    target_size = image_array.shape[:2]
    print(target_size)

    # Konvertiere das Bild in Graustufen
    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print(original_image_gray.shape)

    original_image_resized = cv2.resize(original_image_gray, (target_size[1],target_size[0]))
    print(original_image_resized.shape)


    image_array = np.array(original_image_resized)

    # Vergleiche die Pixel in den beiden Bildern
    difference = original_image_resized - image_array

    print(difference)


def main():
    snipped = cv2.imread('ProcessedImages/Snipped.jpg')
    offset = calcOffset(snipped)

    loaded_Single = cv2.imread('ProcessedImages/2023-12-19_18-31-24_IMG_1.jpg')
    image_fliped= cv2.flip(loaded_Single,1)
    image = invert_with_offset(image_fliped, offset)
    original_image= cv2.imread('Images/Referenz/Analogscan047.jpg')

    compare_images(image, original_image)
    #inver_with_offset_color(test)

if __name__ == '__main__':
    main()
