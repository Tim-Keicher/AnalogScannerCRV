import cv2
import numpy as np

# Define a class for image processing
class ImageProcessing():
    def __init__(self):
        # Initialize threshold values
        self.threshold_value_low = 0#150
        self.threshold_value_high = 255
        self.Image = None
        self.cropedImage = None

    def getImage(self):
        return self.Image

    def setImage(self,img):
        self.Image = img

    # Callback functions for trackbar changes
    def on_trackbar_low(self, val):
        self.threshold_value_low = cv2.getTrackbarPos('Threshold Low', 'Trackbar')

    def on_trackbar_high(self, val):
        self.threshold_value_high = cv2.getTrackbarPos('Threshold High', 'Trackbar')



    # Process the image
    def processImage(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray_image, (7, 7), 1)

        # Define Kernel Size
        kernel = np.ones((35, 35), np.uint8)

        # Apply threshold based on Otsu
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        # Close von Holes, delete blind Spots
        closed_image_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        edges_otsu = cv2.Canny(closed_image_otsu, 30, 150)
        contours_otsu, _ = cv2.findContours(edges_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("otsu", closed_image_otsu)

        # detect corners with the goodFeaturesToTrack function.
        corners = cv2.goodFeaturesToTrack(closed_image_otsu, 4, 0.5, 10)
        corners = np.int0(corners)

        pt_A = corners[0][0]
        pt_B = corners[1][0]
        pt_C = corners[2][0]
        pt_D = corners[3][0]

        width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                 [0, maxHeight - 1],
                                 [maxWidth - 1, maxHeight - 1],
                                 [maxWidth - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        self.cropedImage = cv2.warpPerspective(self.Image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

        cv2.imwrite(f'ProcessedImages/35mmSW_cropped_unwarped.jpg', self.cropedImage)
        cv2.imshow('unwarped', self.cropedImage)



        for contour in contours_otsu:
            x, y, w, h = cv2.boundingRect(contour)
            self.cropedImage = image[y:y+h, x:x+w]
            if w < h:
                self.cropedImage = cv2.rotate(self.cropedImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            #cv2.imwrite("Images/35mmSW_cropped.jpg", self.cropedImage)
#            cv2.imshow("ROI_otsu",  self.cropedImage)
            # Draw green rectangles around contours
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Draw only the first contour

        '''
        # Apply threshold based on trackbar values
        _, thresh_binary = cv2.threshold(blur, self.threshold_value_low, self.threshold_value_high, cv2.THRESH_BINARY)
        # Close von Holes, delete blind Spots
        closed_image = cv2.morphologyEx(thresh_binary, cv2.MORPH_CLOSE, kernel)

        edges_binary = cv2.Canny(closed_image, 30, 150)
        contours_binary, _ = cv2.findContours(edges_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        cv2.imshow('Thresh', closed_image)

        # Find contours
        for contour in contours_binary:
            x, y, w, h = cv2.boundingRect(contour)
            self.cropedImage = image[y:y+h,x:x+w]
            if w < h:
                self.cropedImage = cv2.rotate(self.cropedImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow("ROI",  self.cropedImage)
            # Draw green rectangles around contours
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break  # Draw only the first contour
        '''

        image_path = 'ProcessedImages/35mmSW_cropped_unwarped.jpg'
        image = cv2.imread(image_path)
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        negativ = image.copy()
        cv2.imshow('negative', negativ)
        # Überprüfen, ob das Bild erfolgreich geladen wurde
        if negativ is None:
            print("No Negative Image Exits")
            exit()

        # Kantenerkennung mit Canny
        edges = cv2.Canny(original_image, 180, 230, apertureSize=3)

        # Hough-Linien-Transformation durchführen
        Threshold = 20
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=Threshold, minLineLength=30, maxLineGap=1)

        # Spalten-Summen berechnen
        column_sums = np.sum(negativ, axis=0)

        # Trennpunkte finden (Spalten mit Pixelsumme unter einem Schwellenwert)
        split_points = np.where(column_sums > 25000)[0]

        # Trennlinien in das Originalbild zeichnen
        for split_point in split_points:
            cv2.line(negativ, (split_point, 0), (split_point, negativ.shape[0]), (0, 255, 0),
                     3)  # Trennlinien zeichnen (Farbe: 255)

        # Bilder zwischen den Trennlinien ausschneiden und speichern
        for i in range(len(split_points) - 1):
            start_row = split_points[i]
            end_row = split_points[i + 1]
            print("Start Row {}".format(start_row))
            print("End Row {}".format(end_row))

            if end_row - start_row < 10 or end_row is None:
                snipped = image[:, start_row:end_row]
                if snipped.size > 0:
                    cv2.imwrite('ProcessedImages/Snipped.png', snipped)
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
