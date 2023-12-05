import cv2
import numpy as np

# Bild laden (schwarz-weißes Negativ)
image_path = 'ProcessedImages/35mmSW_cropped.jpg'
image = cv2.imread(image_path)
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

negativ = image.copy()
cv2.imshow('test', negativ)
# Überprüfen, ob das Bild erfolgreich geladen wurde
if negativ is None:
    print(f"Fehler beim Laden des Bildes: {image_path}")
    exit()

# Kantenerkennung mit Canny
edges = cv2.Canny(original_image, 180, 230, apertureSize=3)

# Hough-Linien-Transformation durchführen
Threshold = 30
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=Threshold, minLineLength=30, maxLineGap=1)

# Überprüfen, ob Linien gefunden wurden
if lines is not None:
    prev_line_x = None
    min_distance = 0

    for line in lines:
        x1, _, x2, _ = line[0]

        #if prev_line_x is not None and x1 - prev_line_x >= min_distance:
        #cv2.line(negativ, (x1, 0), (x2, negativ.shape[1]), (255, 255, 255), 2)  # Vertikale Linien zeichnen (Farbe: 0)
        prev_line_x = x2

# Spalten-Summen berechnen
column_sums = np.sum(negativ, axis=0)

# Trennpunkte finden (Spalten mit Pixelsumme unter einem Schwellenwert)
split_points = np.where(column_sums > 12500)[0]

# Trennlinien in das Originalbild zeichnen
for split_point in split_points:
    cv2.line(negativ, (split_point, 0), (split_point, negativ.shape[0]), (0, 255, 0), 3)  # Trennlinien zeichnen (Farbe: 255)
#
# Bilder zwischen den Trennlinien ausschneiden und speichern
for i in range(len(split_points) - 1):
    start_row = split_points[i]
    end_row = split_points[i + 1]
    print("Start Row {}".format(start_row))
    print("End Row {}".format(end_row))

    if end_row-start_row < 10 or end_row is None:
        pass

    else:
        # Bild zwischen den Trennlinien ausschneiden
        cropped_image = image[:, start_row:end_row]

        # Bild speichern (hier als PNG, aber du kannst das Format anpassen)
        cv2.imwrite(f'ProcessedImages/Bild_{i + 1}.png', cropped_image)

# Ergebnis anzeigen
cv2.imshow('Original', original_image)
cv2.imshow('Negatives Bild mit Trennlinien', negativ)
cv2.waitKey(0)
cv2.destroyAllWindows()