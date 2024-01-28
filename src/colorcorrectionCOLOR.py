import cv2
import numpy as np
import skimage

def stretch(plane):
    """
    Stretch the pixel intensity values of a given image plane.

    Parameters:
    - plane: Input image plane (single color channel)

    Returns:
    - Stretched image plane
    """

    # Calculate the 1st and 99th percentiles to determine the intensity range for stretching
    imin = np.percentile(plane, 1)
    imax = np.percentile(plane, 99)

    # Stretch the pixel intensity values of the image plane to the [0, 1] range
    plane = (plane - imin) / (imax - imin)

    return plane

def invert_with_offset(img, showImage=False):
    """
    Invert the color channels of the image considering an offset value.

    Parameters:
    - img: Input image
    - showImage: Indicates whether to display the inverted image (Default: False)

    Returns:
    - Normalized inverted RGB image
    """

    # Split the input image into color channels
    (bneg, gneg, rneg) = cv2.split(img)

    # Apply the stretch function to each color channel and invert the color channels
    b = 1 - np.clip(stretch(bneg), 0, 1)
    g = 1 - np.clip(stretch(gneg), 0, 1)
    r = 1 - np.clip(stretch(rneg), 0, 1)

    # Apply the gamma correction algorithm to each color channel
    # Here, the gamma value is adjusted based on the mean value of the color channel and the offset
    b = skimage.exposure.adjust_gamma(b, gamma=b.mean()/0.65)
    g = skimage.exposure.adjust_gamma(g, gamma=g.mean()/0.25)
    r = skimage.exposure.adjust_gamma(r, gamma=r.mean()/0.5)

    # Merge the color channels back into an RGB image
    inverted_rgb = cv2.merge([b, g, r])

    # Normalize the inverted RGB values to the range [0, 255] and convert the data type to uint8
    normalized_inverted_rgb = np.multiply(inverted_rgb, 255).astype('uint8')

    # Display the inverted image if showImage is set to True
    if showImage:
        cv2.imshow('Inverted Image with Offset', normalized_inverted_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return normalized_inverted_rgb
