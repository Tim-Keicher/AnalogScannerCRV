import customtkinter as ctk

class ImageFrame(ctk.CTkFrame):
    """
    A custom tkinter frame designed to display a dynamic number of images in a grid layout.
    The layout adjusts based on the count of images, arranging them in 1xN (N up to 3), 2x2, 
    or 2x3 formats.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initializes the ImageFrame with a list to store image labels.

        Parameters:
            parent (tkinter.Widget): The parent widget.
            *args, **kwargs: Additional arguments and keyword arguments to pass to the CTkFrame constructor.
        """
        ctk.CTkFrame.__init__(self, parent, *args, **kwargs)
        self.image_labels = []  # List to store image label

    def update_images(self, images):
        """
        Clears existing image labels and dynamically creates new labels based on the count of images.
        The layout adjusts to accommodate up to 6 images.
        Calculates the appropriate grid size and image size based on the number of images.
        Uses CTkImage for compatibility and prevents images from being garbage collected.

        Parameters:
            images (list): A list of images to be displayed.
        """
        # Clear existing labels
        for label in self.image_labels:
            label.destroy()

        # Determine the number of rows and columns based on the image count
        image_count = len(images)
        rows, columns = self.calculate_rows_columns(image_count)

        # Get image size corresponds to the number of images to be plotted
        if image_count < 3:
            img_size = [840 / image_count, 580 / image_count]
        else:
            img_size = [840 / 3, 580 / 3]

        # Create and add image labels based on the number of images
        for i, image in enumerate(images):
            #image.thumbnail((100, 100))  # Resize image to fit in the label
            ctk_image = ctk.CTkImage(image, size=img_size)  # Use ctk.CTkImage for better compatibility

            # Calculate row and column indices based on the layout
            row_index = i // columns
            column_index = i % columns
            
            # Create a new label for each image
            label = ctk.CTkLabel(self, image=ctk_image, text="")
            label.grid(row=row_index, column=column_index, padx=10, pady=(10, 10))
            label.image = ctk_image  # Keep a reference to prevent image from being garbage collected
            
            # Add the label to the list
            self.image_labels.append(label)

    def calculate_rows_columns(self, image_count):
        """
        Determines the layout based on the count of images.
        Returns the number of rows and columns suitable for up to 6 images.

        Parameters:
            image_count (int): The count of images.

        Returns:
            tuple: A tuple representing the number of rows and columns for the grid layout.
        """
        if image_count <= 3:
            return 1, image_count
        elif image_count == 4:
            return 2, 2
        elif image_count <= 6:
            return 2, 3
        else:
            # Handle additional cases as needed
            return 0, 0
