import numpy as np
import cv2
import skimage

im = cv2.imread("Images/test_internet.png")
im = cv2.imread("Images/test_color_img.png")

(bneg, gneg, rneg) = cv2.split(im)

def stretch(plane):
    imin = np.percentile(plane, 1)
    imax = np.percentile(plane, 99)
    plane = (plane - imin) / (imax - imin)
    return plane

b = 1 - np.clip(stretch(bneg), 0, 1)
g = 1 - np.clip(stretch(gneg), 0, 1)
r = 1 - np.clip(stretch(rneg), 0, 1)

# Initialwerte für die Slider
slider_b = 1.0
slider_g = 1.0
slider_r = 1.0

def update_slider_b(value):
    global slider_b
    slider_b = value / 100.0
    print("[slider b] " + str(slider_b))

def update_slider_g(value):
    global slider_g
    slider_g = value / 100.0
    print("[slider g] " + str(slider_g))

def update_slider_r(value):
    global slider_r
    slider_r = value / 100.0
    print("[slider r] " + str(slider_r))

# OpenCV-Fenster erstellen
cv2.namedWindow('im bgr')

# Schieberegler erstellen und an das Fenster anhängen
cv2.createTrackbar('Slider b', 'im bgr', int(slider_b * 100), 100, update_slider_b)
cv2.createTrackbar('Slider g', 'im bgr', int(slider_g * 100), 100, update_slider_g)
cv2.createTrackbar('Slider r', 'im bgr', int(slider_r * 100), 100, update_slider_r)

while True:
    # Bild bearbeiten
    b_adjusted = skimage.exposure.adjust_gamma(b, gamma=b.mean() / slider_b)
    g_adjusted = skimage.exposure.adjust_gamma(g, gamma=g.mean() / slider_g)
    r_adjusted = skimage.exposure.adjust_gamma(r, gamma=r.mean() / slider_r)
    
    # Kanäle zusammenführen
    bgr = cv2.merge([b_adjusted, g_adjusted, r_adjusted])

    # Bild im Fenster anzeigen
    cv2.imshow('im bgr', bgr)

    # Auf Tastendruck warten (ESC zum Beenden)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Fenster schließen
cv2.destroyAllWindows()
