import cv2
import numpy as np

def calcOffset(snipped, verbose=False):
    img = cv2.cvtColor(snipped, cv2.COLOR_BGR2GRAY)

    average_value = int(img.mean())
    if verbose:
        print(f"Average Pixelvalue: {average_value}")
    return average_value


def invert_with_offset(img, offset, showImage=False):
    # In Graustufen umwandeln, falls das Bild nicht bereits in Graustufen ist
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # white balancing
    white_balance_img = cv2.add(img, offset)

    # Inversion der Farben
    inverted_white_balance_img = cv2.bitwise_not(white_balance_img)
    MaxValue = np.max(inverted_white_balance_img)
    factor = 255/MaxValue

    inverted_white_balance_img = np.multiply(inverted_white_balance_img, factor).astype('uint8')
    data = inverted_white_balance_img

    if showImage:
        cv2.imshow('Inverted with Offset', data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return inverted_white_balance_img


