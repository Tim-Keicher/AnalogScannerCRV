# Design refer to example on: https://github.com/TomSchimansky/CustomTkinter

import customtkinter as ctk
import os
import cv2
from PIL import Image
from tkinter import filedialog

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # Set variables
        self.load_location_path:str = None
        self.save_location_path:str = None

        # __init__ function for class Tk
        ctk.CTk.__init__(self, *args, **kwargs)
        
        # Set the width and height for the window
        self.geometry(f"{1100}x{580}")
        #self.after(0, lambda:self.state('zoomed'))

        # Set window title
        self.title('Analog Scanner')

        # Set window bitmap icon
        self.after(201, lambda:self.iconbitmap('imagesGUI\HHN_Logo_D_weiss.ico'))

        # Set GUI image path
        gui_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "imagesGUI")
        self.logo_image = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "HHN_Logo_D_weiss.png")), size=(138, 58))
        self.play_icon = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "play_white_icon.jpg")), size=(20, 20))
        self.save_icon = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "save_white_icon.jpg")), size=(20, 20))
        self.load_icon = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "folder_white_icon.jpg")), size=(20, 20))
        
        #----------------------------------------------------------------------------------------------------
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=7, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(self.sidebar_frame, text=None, image=self.logo_image,
                                                             compound="center", corner_radius=8) #fg_color="#b5b5b5", corner_radius=8)
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.sidebar_camera_image = ctk.CTkOptionMenu(self.sidebar_frame, values=["Image", "Camera"], command=self.sidebar_cam_img_event)
        self.sidebar_camera_image.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_load = ctk.CTkButton(self.sidebar_frame, text="Load Location", image=self.load_icon, anchor='w', command=self.sidebar_btn_load_event)    # need load button as soon as image is selected
        self.sidebar_load.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_camera_port = ctk.CTkOptionMenu(self.sidebar_frame, values=self.get_connected_camera_ports(), command=self.sidebar_cam_port_event) # need port dropdown as soon as camera is selected
        #self.sidebar_camera_port.grid(row=2, column=0, padx=20, pady=10)   # show sidebar_load button first

        self.sidebar_img_format = ctk.CTkOptionMenu(self.sidebar_frame, values=["Small Format", "Middle Format"], command=self.sidebar_format_event)
        self.sidebar_img_format.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_save_cb = ctk.CTkCheckBox(self.sidebar_frame, text="Save Images", command=self.sidebar_cb_save_event)
        self.sidebar_save_cb.grid(row=4, column=0, padx=20, pady=10)

        self.sidebar_save_location = ctk.CTkButton(self.sidebar_frame, text="Save Location", image=self.save_icon, anchor='w', command=self.sidebar_btn_save_event)
        self.sidebar_save_location.grid(row=5, column=0, padx=20, pady=10)

        self.sidebar_process = ctk.CTkButton(self.sidebar_frame, text="Process", image=self.play_icon, anchor='w', command=self.sidebar_btn_process_event)
        self.sidebar_process.grid(row=6, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))

        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))

        # Set default values
        self.sidebar_camera_image.set("Image")
        self.sidebar_img_format.set("Small Format")
        self.sidebar_save_cb.deselect()
        self.sidebar_save_location.configure(state="disabled")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    #----------------------------------------------------------------------------------------------------
    # Sidebar callback functions
    def sidebar_cam_img_event(self, option:str):
        if option == "Camera":
            self.sidebar_load.grid_forget()
            self.sidebar_camera_port.grid(row=2, column=0, padx=20, pady=10)
            self.start_webcam()
        else:
            self.sidebar_camera_port.grid_forget()
            self.sidebar_load.grid(row=2, column=0, padx=20, pady=10)
            self.stop_webcam()

    def sidebar_btn_load_event(self):
        self.load_location_path = filedialog.askopenfilename(initialdir='Images', title='Select a image!', defaultextension='.png', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("All Files", "*.*")])
        print(self.load_location_path)

    def sidebar_cam_port_event(self, option:str):
        print(option)
        
    def sidebar_format_event(self, option:str):
        print(option)

    def sidebar_cb_save_event(self):
        if self.sidebar_save_cb.get():
            self.sidebar_save_location.configure(state="enabled")
        else:
            self.sidebar_save_location.configure(state="disabled")

    def sidebar_btn_save_event(self):
        self.save_location_path = filedialog.asksaveasfilename(initialdir='Images', title='Select save location!', defaultextension='.png', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("All Files", "*.*")])
        print(self.save_location_path)

    def sidebar_btn_process_event(self):
        print("sidebar_btn_process_event click")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    #----------------------------------------------------------------------------------------------------
    # Camera functions
    def start_webcam(self):
        # Start the webcam
        self.video_capture = cv2.VideoCapture(0)

        # Create a Tkinter Label to display the camera image
        self.camera_label = ctk.CTkLabel(self, text="")
        self.camera_label.grid(row=0, column=1, rowspan=7, padx=20, pady=20)

        # Start the update function
        self.update_camera()

    # Update the camera image in the Tkinter window
    def update_camera(self):
        _, frame = self.video_capture.read()
        if frame is not None:
            # OpenCV returns images in BGR format, so we need to convert it to RGB format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the image to a Tkinter PhotoImage
            ctk_image = ctk.CTkImage(Image.fromarray(rgb_image), size=(850, 500))
            # Update the image in the Tkinter Label
            self.camera_label.configure(image=ctk_image)
            self.camera_label.image = ctk_image
        # Update the image again after a certain time (here: 30 milliseconds)
        self.after(30, self.update_camera)

    def stop_webcam(self):
        # Stop the update function and turn off the camera
        self.after_cancel(self.update_camera)
        self.video_capture.release()
        self.camera_label.destroy()

    def get_connected_camera_ports(self):
        connected_ports = []

        # Start with index 0 and increment until no camera is found
        index = 0
        while True:
            # Try to open the camera with the current index
            cap = cv2.VideoCapture(index)
            
            # Check if the camera is opened successfully
            if not cap.isOpened():
                break

            # Release the camera capture object
            cap.release()

            # Append the current index to the list of connected ports
            connected_ports.append("port-" + str(index))

            # Increment the index for the next iteration
            index += 1

        return connected_ports
        

if __name__ == '__main__':
    app = App()
    app.mainloop()