import cv2
import numpy as np

'''
Bei der Manipulation von Farben und Helligkeit kann die Verwendung des HSV-Farbraums einige Vorteile bieten. HSV steht für Hue (Farbton), Saturation (Sättigung) und Value (Wert). Im Gegensatz zum RGB-Farbraum, in dem Farben als Kombination von Rot, Grün und Blau dargestellt werden, ermöglicht der HSV-Farbraum eine intuitive Steuerung der Farbe, Sättigung und Helligkeit unabhängig voneinander.

Helligkeitseinstellung: Im HSV-Farbraum ist der Helligkeitskanal (Value) separat, was die Anpassung der Helligkeit erleichtert. Wenn du nur die Helligkeit ändern möchtest, ohne die Farben zu beeinflussen, ist die Arbeit im HSV-Farbraum einfacher.

Invertierung: Bei der Invertierung von Farben kannst du den Farbton (Hue) beibehalten und nur die Sättigung und Helligkeit umkehren. Dies kann in HSV einfacher sein als in RGB, wo eine direkte Invertierung aller Kanäle (Rot, Grün, Blau) komplizierter sein kann und nicht das gewünschte Ergebnis liefert.

Unabhängige Kanäle: Im HSV-Farbraum sind Farbton, Sättigung und Helligkeit unabhängige Kanäle. Dies ermöglicht es, verschiedene Farb- und Helligkeitsanpassungen getrennt durchzuführen, was oft flexibler ist als bei RGB.

Für spezifische Bildmanipulationen kann die Wahl zwischen RGB und HSV von der Natur der gewünschten Änderungen abhängen. In diesem Fall, wenn du die Helligkeit anpassen und Farben invertieren möchtest, bietet der HSV-Farbraum eine effiziente Methode, um dies zu erreichen, ohne die Farbkanäle direkt zu verändern, was zu unerwünschten Ergebnissen führen könnte.
'''


def calcOffset(snipped, verbose=False):
    # Convert to HSV Colorspace
    img_hsv = cv2.cvtColor(snipped, cv2.COLOR_BGR2HSV)
    # Calculate average in hsv channel
    average_value = int(img_hsv[:,:,2].mean())
    if verbose:
        print(f"Average pixel value: {average_value}")
    return average_value


def invert_with_offset(img, offset, showImage=False):
    # Invertiere die Farbkanäle
    inverted_rgb = cv2.bitwise_not(img)

    # Korrigiere die Helligkeit
    inverted_rgb = cv2.add(inverted_rgb, offset)

    # Checke die Werte auf den Bereich 0-255
    normalized_inverted_rgb = np.clip(inverted_rgb, 0, 255).astype(np.uint8)
    normalized_inverted_rgb=cv2.cvtColor(normalized_inverted_rgb, cv2.COLOR_BGR2RGB)
    if showImage:
        cv2.imshow('Invertiert mit Offset', normalized_inverted_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return normalized_inverted_rgb
