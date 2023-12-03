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

# Konturen im Bild finden
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Negatives Bild erstellen und zuschneiden
negatives = np.zeros_like(image)

for contour in contours:
    # Rechteck um jede Kontur zeichnen
    x, y, w, h = cv2.boundingRect(contour)

    # Grünen Rahmen um die Kontur zeichnen
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Ausschnitt des Bildes extrahieren
    tile = image[y:y + h, x:x + w]

    # Bild negieren
    negative_tile = cv2.bitwise_not(tile)

    # Negatives Bild aktualisieren
    negatives[y:y + h, x:x + w] = negative_tile


# Summe der Pixelwerte in den Spalten berechnen
column_sums = np.sum(negatives, axis=0)
print(column_sums)

# Schwellenwert für die Identifikation der Spalten festlegen
threshold = 255 * (negatives.shape[0] - 1)

# Spalten mit Pixelsumme über dem Schwellenwert identifizieren
marked_columns = np.where(column_sums >= threshold)[0]

# Lila Farbcode für markierte Bereiche
lila_farbcode = (128, 0, 128)

# Markierte Spalten durch Lila ersetzen
for col in marked_columns:
    image[:, col, :] = lila_farbcode

'''# Hough-Linien-Transformation durchführen
lines = cv2.HoughLinesP(edges, 1, np.pi, threshold=50, minLineLength=70, maxLineGap=10)

# Überprüfen, ob Linien gefunden wurden
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)'''

# Ursprüngliches und negatives Bild nebeneinander anzeigen
result = np.hstack((image, negatives))

'''# Index der schwarzen Pixel finden
black_pixels = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)

# Grüne Farbe für schwarze Pixel setzen
result[black_pixels] = [128, 0, 128]  # Grüner Farbwert: [0, 255, 0]

#print(result)'''

# Fenster mit dem Ergebnis anzeigen
cv2.namedWindow('Original und Negativ', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original und Negativ', 1080, 180)
cv2.imshow('Original und Negativ', result)


# Warte darauf, dass eine Taste gedrückt wird, und schließe dann das Fenster
cv2.waitKey(0)
cv2.destroyAllWindows()
