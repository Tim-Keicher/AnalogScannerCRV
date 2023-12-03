import cv2
import numpy as np

# Bild laden
image_path = 'Images/35mmSW_cropped.jpg'
image = cv2.imread(image_path)

# Überprüfen, ob das Bild erfolgreich geladen wurde
if image is None:
    print(f"Fehler beim Laden des Bildes: {image_path}")
    exit()

# Grayscale-Version des Bildes erstellen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Kantenerkennung mit Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough-Linien-Transformation durchführen
lines = cv2.HoughLinesP(edges, 1, np.pi, threshold=50, minLineLength=50, maxLineGap=10)

# Überprüfen, ob Linien gefunden wurden
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Ergebnis anzeigen
cv2.imshow('Bild mit markierter vertikaler Linie', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
