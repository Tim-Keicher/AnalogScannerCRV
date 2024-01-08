import os.path

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import image_processing as iP


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Negative Analog Transformer")
        self.frame = tk.Frame(self.root)
        # Own Icon
        self.root.iconbitmap('../Images/Referenz/HHN.ico')
        self.frame.pack(padx=20, pady=20)

        self.process_options = ["Colored", "Black&White"]
        self.selected_process = tk.StringVar()
        self.selected_process.set(self.process_options[0])

        self.process_dropdown = tk.OptionMenu(self.frame, self.selected_process, *self.process_options)
        self.process_dropdown.config(bg="white", fg="black")
        self.process_dropdown.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        self.camera_button = tk.Button(self.frame, text="Camera", command=self.capture_from_camera,
                                       activebackground="lightblue")
        self.camera_button.grid(row=1, column=0, padx=5, pady=5)

        self.image_button = tk.Button(self.frame, text="Image from path", command=self.load_image_from_path,
                                      activebackground="lightblue")
        self.image_button.grid(row=1, column=1, padx=5, pady=5)

        self.detect_button = tk.Button(self.frame, text="Detect Single Images", command=self.detect_images,
                                       activebackground="lightblue")
        self.detect_button.grid(row=2, column=0, padx=5, pady=5)

        self.invert_button = tk.Button(self.frame, text="Invert Image", command=self.invert_image,
                                       activebackground="lightblue")
        self.invert_button.grid(row=2, column=1, padx=5, pady=5)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=20, pady=20)

        # Init Image Processing
        self.imPro = iP.ImageProcessing()
        self.Image = None
        self.cropped_img = None

    def capture_from_camera(self):
        # Funktion für die Kameranutzung (muss implementiert werden)
        pass

    def load_image_from_path(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.Image = cv2.imread(file_path)
            self.imPro.showImg(window_name='Loaded Image, press any key to close', img=self.Image)
            # image = Image.fromarray(image)
            # photo = ImageTk.PhotoImage(image=image)
            # self.image_label.configure(image=photo)
            # self.image_label.image = photo

    def detect_images(self):
        if self.Image is not None:
            self.cropped_img = self.imPro.cutStrip(img=self.Image)
            self.imPro.showImg(window_name='Cropped Image, press any key to close', img=self.cropped_img)
        else:
            print(f'[WARNING] self.Image is {self.Image}')
            pass

    def invert_image(self):
        # Funktion für das Bildinvertierung
        pass


def main():
    root = tk.Tk()
    root.iconbitmap('../Images/Referenz/HHN.ico')
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
