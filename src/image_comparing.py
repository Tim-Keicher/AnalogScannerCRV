import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def compare_images():

    original_image = cv2.imread('Images/Referenz/Analogscan047.jpg')
    inverted_image = cv2.imread('Saves/clahe_contrast11-2024-01-24_17-24-37.jpg')

    inverted_image_gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    original_image_resized = cv2.resize(original_image_gray, (inverted_image_gray.shape[1], inverted_image_gray.shape[0]))

    border_size = 10
    crp_inverted_image_gray = inverted_image_gray[border_size:-border_size, border_size:-border_size]
    crp_original_image_resized = original_image_resized[border_size:-border_size, border_size:-border_size]

    diff = np.array(crp_original_image_resized) - np.array(crp_inverted_image_gray)

    similarity_index = ssim(crp_original_image_resized, crp_inverted_image_gray, data_range=crp_inverted_image_gray.max()-crp_inverted_image_gray.min())
    mean_root_square = mse(crp_original_image_resized, crp_inverted_image_gray)

    print("[SSIM] " + str(similarity_index))
    print("[MSE] " + str(mean_root_square))
    print("[Max] "+ str(np.max(diff)))
    print("[Min] " + str(np.min(diff)))
    print("[Diff] " + str(np.sum(diff)))

    #output_file_path = 'pixel_values.txt'

    # Schreibe die Pixelwerte in die Datei
    #np.savetxt(output_file_path, diff, fmt='%d', delimiter=', ')

    cv2.imwrite('cmp_diff_img.png', diff)


import colorcorectionBW as ccBW

if __name__ == '__main__':
    #img_original = cv2.imread('Images/Referenz/Analogscan046.jpg')
    #img_inverted = cv2.imread('Saves/Bilder5-2024-01-21_11-59-19-.jpg')

    #cv2.imshow("img_original", snipped)
    # cv2.imshow("img_inverted", img_inverted)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    compare_images()