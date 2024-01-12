import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog

class ImageFrame(ctk.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkFrame.__init__(self, parent, *args, **kwargs)

        # Create a list to store image labels
        self.image_labels = []

    def update_images(self, image_paths):
        # Clear existing labels
        for label in self.image_labels:
            label.destroy()

        # Determine the number of rows and columns based on the image count
        image_count = len(image_paths)
        rows, columns = self.calculate_rows_columns(image_count)

        # Get image size corresponds to the number of images to be plotted
        if image_count < 3:
            img_size = [840 / image_count, 580 / image_count]
        else:
            img_size = [840 / 3, 580 / 3]

        # Create and add image labels based on the number of images
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            #image.thumbnail((100, 100))  # Resize image to fit in the label
            ctk_image = ctk.CTkImage(image, size=img_size)  # Use ctk.CTkImage for better compatibility

            # Calculate row and column indices based on the layout
            row_index = i // columns
            column_index = i % columns
            
            # Create a new label for each image
            label = ctk.CTkLabel(self, image=ctk_image, text="")
            label.grid(row=row_index, column=column_index, padx=10, pady=(10, 10))
            label.image = ctk_image  # Keep a reference to prevent image from being garbage collected
            label.bind("<Button-1>", lambda event, path=image_path: self.on_image_click(path))
            
            # Add the label to the list
            self.image_labels.append(label)

    def calculate_rows_columns(self, image_count):
        # Determine the layout based on the image count
        if image_count <= 3:
            return 1, image_count
        elif image_count == 4:
            return 2, 2
        elif image_count <= 6:
            return 2, 3
        else:
            # Handle additional cases as needed
            return 0, 0

    def on_image_click(self, path):
        # Handle image click event (you can customize this function)
        print("Image clicked:", path)
