import os.path

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import image_processing as iP


class ImageProcessorApp:
    def __init__(self, root):
        # Variables
        self.name_colored = "Colored"
        self.name_bw = "Black&White"
        self.name_dia = "Diapositiv"

        # Define root window of GUI
        self.root = root
        self.root.title("Negative Analog Transformer")
        self.frame = tk.Frame(self.root)
        # Own Icon
        self.root.iconbitmap('../Images/Referenz/HHN.ico')
        self.frame.pack(padx=20, pady=20)

        # Define dropdown menu
        self.process_options = [self.name_colored, self.name_bw, self.name_dia]
        self.selected_process = tk.StringVar()
        self.selected_process.set(self.process_options[0])

        # generate dropdown menu
        self.process_dropdown = tk.OptionMenu(self.frame, self.selected_process, *self.process_options)
        self.process_dropdown.config(bg="white", fg="black")
        self.process_dropdown.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Generate Buttons
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

        # define image Position in GUI window
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=20, pady=20)

        # Init Image Processing
        self.imPro = iP.ImageProcessing()
        self.Image = None
        self.cropped_img = None
        self.pathType = None
        self.popup_win = None


    def capture_from_camera(self):
        # Implementierung der Funktion für die Kameranutzung
        pass

    def load_image_from_path(self):
        self.popup()
        dataset = []

        if self.pathType == "File":
            file_path = filedialog.askopenfilename(title="Chose Image", initialdir="/", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("All Files", "*.*")])
            print(file_path)
            if file_path != "":
                self.Image = cv2.imread(file_path)
                self.imPro.showImg(window_name='Loaded Image, press any key to close', img=self.Image)
                dataset.append(self.Image)

        elif self.pathType == "Dir":
            folder_path = filedialog.askdirectory(title="Chose Folder with Images", initialdir="/")

            if folder_path:  # Check if it is folder path
                objects_in_folder = os.listdir(folder_path)
                for obj in objects_in_folder:
                    print(obj)
                    # ToDo : check for image file format
                    dataset.append(cv2.imread(folder_path+obj))
            else:
                print("[WARNING] Selected object is no folder")
                pass

        else:
            print(f'[Warning] pathType {self.pathType} unknown')
            pass

        print(f'[INFO] Loaded Images: {len(dataset)}')


    def detect_images(self):
        if self.Image is not None:
            self.cropped_img = self.imPro.cutStrip(img=self.Image)
            self.imPro.showImg(window_name='Cropped Image, press any key to close', img=self.cropped_img)
        else:
            print(f'[WARNING] self.Image is {self.Image}')
            pass

    def invert_image(self):
        mode = self.get_dropdown()
        print(f"[INFO] mode {mode} have been selected")
        # Inverting images based on the type of film
        if mode == self.name_bw:
            pass

        elif mode == self.name_colored:
            pass

        elif mode == self.name_dia:
            pass

    def get_dropdown(self):
        # Return the value of the dropdown menu
        selected_value = self.selected_process.get()
        return selected_value

    def antwort_auswerten(self, answer):
        if answer == 'File':
            self.pathType = 'File'
        else:
            self.pathType = 'Dir'

        self.popup_win.destroy()

    def popup(self):
        self.popup_win = tk.Toplevel()  # Änderung: Verwendung von Toplevel für das Popup
        self.popup_win.wm_title('Define Loaded File')
        label = tk.Label(self.popup_win, text='What do you want to load?')
        label.pack()

        button_file = tk.Button(self.popup_win, text='File', command=lambda: self.antwort_auswerten('File'))
        button_file.pack()

        button_dir = tk.Button(self.popup_win, text='Directory', command=lambda: self.antwort_auswerten('Directory'))
        button_dir.pack()

        # Verwendung von wait_window, um auf das Schließen des Popup-Fensters zu warten
        self.root.wait_window(self.popup_win)

def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
