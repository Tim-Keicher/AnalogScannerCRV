import cv2
import numpy as np

# Bild laden
image_path = "../Images/test_color_img.png"
snipped_path = "../Images/Color_snipped.JPG"
image = cv2.imread(image_path)
snipped = cv2.imread(snipped_path)

def adjust_contrast_and_balance(img, alpha=1.2, beta=1, offset=None, showImage=False):
    # In Graustufen umwandeln, falls das Bild nicht bereits in Graustufen ist
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # White balancing
    if offset is not None:
        img_gray = cv2.add(img_gray, offset)

    # Inversion der Farben
    inverted_img = cv2.bitwise_not(img_gray)

    # Adjust contrast and brightness
    adjusted_img = cv2.convertScaleAbs(inverted_img, alpha=alpha, beta=beta)

    if showImage:
        combined_image = np.hstack((img, inverted_img, adjusted_img))
        cv2.imshow('combined_image', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return adjusted_img


def calcOffset(snipped, verbose=False):
    # Check if the image is single-channel (already grayscale)
    if len(snipped.shape) == 2:
        img = snipped
    else:
        img = cv2.cvtColor(snipped, cv2.COLOR_BGR2GRAY)

    average_value = int(img.mean())
    if verbose:
        print(f"Average Pixelvalue: {average_value}")
    return average_value


def main():
    if image is None:
        print(f"Fehler beim Laden des Bildes: {image_path}")
        exit()
    if snipped is None:
        print(f"Fehler beim Laden des Bildes: {snipped_path}")
        exit()

    # Farbkanäle extrahieren
    b, g, r = cv2.split(image)
    snipped_b, snipped_g, snipped_r = cv2.split(snipped)

    average_value_b = 255 - calcOffset(snipped_b)
    print(average_value_b)
    average_value_g = 255 - calcOffset(snipped_g)
    print(average_value_g)
    average_value_r = 255 - calcOffset(snipped_r)
    print(average_value_r)

    adjusted_image_b = adjust_contrast_and_balance(b, offset=average_value_b, showImage=False)
    adjusted_image_g = adjust_contrast_and_balance(g, offset=average_value_g, showImage=False)
    adjusted_image_r = adjust_contrast_and_balance(r, offset=average_value_r, showImage=False)

    adjusted_image = cv2.merge([adjusted_image_b, adjusted_image_g, adjusted_image_r])
    cv2.imshow('Adjusted Image', adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()