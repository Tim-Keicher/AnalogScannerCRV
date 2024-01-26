import cv2
import numpy as np

def calcOffset(snipped, verbose=False):
    """Calculate the average pixel value in a grayscale image.

    Args:
        snipped (numpy.ndarray): Grayscale image.
        verbose (boolean): Print average pixel value if True.

    Returns:
        int: Average pixel value.

    Notes:
        This method converts the input image to grayscale and calculates the average pixel value.

    Raises:
        None
    """
    # Convert the input image to grayscale
    img = cv2.cvtColor(snipped, cv2.COLOR_BGR2GRAY)

    # Calculate the average pixel value
    average_value = int(img.mean())

    # Print the average pixel value if verbose is True
    if verbose:
        print(f"Average Pixelvalue: {average_value}")

    return average_value

def invert_with_offset(img, offset, showImage=False):
    """Invert the colors of an image with a specified offset and perform white balancing.

    Args:
        img (numpy.ndarray): Input image.
        offset (int): Offset value for white balancing.
        showImage (boolean): Display the inverted image if True.

    Returns:
        numpy.ndarray: Inverted and white-balanced image.

    Notes:
        This method converts the input image to grayscale if it is not already in grayscale.
        It performs white balancing by adding the specified offset to the pixel values.
        The colors are inverted, and the image is adjusted to have a maximum pixel value of 255.

    Raises:
        None
    """
    # Convert the input image to grayscale if not already in grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # White balancing by adding the offset to pixel values
    white_balance_img = cv2.add(img, offset)

    # Invert the colors
    inverted_white_balance_img = cv2.bitwise_not(white_balance_img)

    # Adjust the image to have a maximum pixel value of 255
    MaxValue = np.max(inverted_white_balance_img)
    factor = 255 / MaxValue
    inverted_white_balance_img = np.multiply(inverted_white_balance_img, factor).astype('uint8')
    data = inverted_white_balance_img

    # Display the inverted image if showImage is True
    if showImage:
        cv2.imshow('Inverted with Offset', data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return inverted_white_balance_img
