# Design refer to example on: https://github.com/TomSchimansky/CustomTkinter

import customtkinter as ctk
import os
import cv2
import re
from PIL import Image
from tkinter import filedialog

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    """
    The main application class for the Analog Scanner.

    Attributes:
        load_location_path (str): Path to the loaded image file.
        save_location_path (str): Path to save the processed image.
        current_camera_port (str): Currently selected camera port.

    Methods:
        __init__(self, *args, **kwargs): Initializes the application window and sets up the GUI elements.
        sidebar_cam_img_event(self, option: str): Handles events when switching between Image and Camera modes.
        sidebar_btn_load_event(self): Handles events when the "Load Location" button is clicked.
        sidebar_cam_port_event(self, option: str): Handles events when selecting a camera port from the dropdown.
        sidebar_format_event(self, option: str): Handles events when selecting image format from the dropdown.
        sidebar_cb_save_event(self): Handles events when the "Save Images" checkbox is clicked.
        sidebar_btn_save_event(self): Handles events when the "Save Location" button is clicked.
        sidebar_btn_process_event(self): Handles events when the "Process" button is clicked.
        change_appearance_mode_event(self, new_appearance_mode: str): Handles events when changing appearance mode.
        change_scaling_event(self, new_scaling: str): Handles events when changing UI scaling.
        start_webcam(self, port: int = None): Starts the webcam and updates the camera image in the GUI.
        update_camera(self): Continuously updates the camera image in the GUI.
        stop_webcam(self): Stops the webcam and clears the camera image from the GUI.
        get_connected_camera_ports(self): Gets a list of connected camera ports.

    Note:
        This class inherits from the ctk.CTk class provided by the customtkinter library.
    """
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # Set variables
        self.load_location_path:str = None
        self.save_location_path:str = None
        self.current_camera_port:str = None

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
        #self.sidebar_camera_port.set("port-1")
        self.sidebar_img_format.set("Small Format")
        self.sidebar_save_cb.deselect()
        self.sidebar_save_location.configure(state="disabled")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    #----------------------------------------------------------------------------------------------------
    # Sidebar callback functions
    def sidebar_cam_img_event(self, option:str):
        """
        Handles events when switching between Image and Camera modes.

        Parameters:
            option (str): The selected mode, either "Image" or "Camera".
        """
        if option == "Camera":
            self.sidebar_load.grid_forget()
            self.sidebar_camera_port.grid(row=2, column=0, padx=20, pady=10)

            # Check available cameras again and adapte visible ports
            val = self.get_connected_camera_ports()
            self.sidebar_camera_port.configure(values=val)
            if self.sidebar_camera_port.get() == "no cameras":
                self.sidebar_camera_port.set(val[0])    # update only, if no camera was detected before

            # Get current port from option menu
            self.current_camera_port = self.sidebar_camera_port.get()

            # Get camera port from string
            match = re.search(r'\bport-(\d+)\b', self.current_camera_port)
            if match:
                self.start_webcam(int(match.group(1)))
        else:
            self.sidebar_camera_port.grid_forget()
            self.sidebar_load.grid(row=2, column=0, padx=20, pady=10)
            self.stop_webcam()

    def sidebar_btn_load_event(self):
        """
        Handles events when the "Load Location" button is clicked.
        Opens a file dialog for selecting image files.
        """
        self.load_location_path = filedialog.askopenfilename(initialdir='Images', title='Select a image!', multiple=True, defaultextension='.png', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("All Files", "*.*")])
        for img in self.load_location_path:
            print(img)

    def sidebar_cam_port_event(self, option:str):
        """
        Handles events when selecting a camera port from the dropdown.

        Parameters:
            option (str): The selected camera port.
        """
        if self.current_camera_port is not option:
            self.current_camera_port = option

            # stop current webcam first
            self.stop_webcam()

            # Switch camera
            match = re.search(r'\bport-(\d+)\b', self.current_camera_port)
            if match:
                self.start_webcam(int(match.group(1)))
        
    def sidebar_format_event(self, option:str):
        """
        Handles events when selecting image format from the dropdown.

        Parameters:
            option (str): The selected image format.
        """
        print(option)

    def sidebar_cb_save_event(self):
        """
        Handles events when the "Save Images" checkbox is clicked.
        Enables or disables the "Save Location" button based on checkbox state.
        """
        if self.sidebar_save_cb.get():
            self.sidebar_save_location.configure(state="enabled")
        else:
            self.sidebar_save_location.configure(state="disabled")

    def sidebar_btn_save_event(self):
        """
        Handles events when the "Save Location" button is clicked.
        Opens a file dialog for selecting the location to save processed images.
        """
        self.save_location_path = filedialog.asksaveasfilename(initialdir='Images', title='Select save location!', defaultextension='.png', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("All Files", "*.*")])
        print(self.save_location_path)

    def sidebar_btn_process_event(self):
        """
        Handles events when the "Process" button is clicked.
        Initiates the image processing or camera capturing based on the selected mode.
        """
        print("sidebar_btn_process_event click")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        """
        Handles events when changing appearance mode.

        Parameters:
            new_appearance_mode (str): The selected appearance mode.
        """
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        """
        Handles events when changing UI scaling.

        Parameters:
            new_scaling (str): The selected UI scaling percentage.
        """
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    #----------------------------------------------------------------------------------------------------
    # Camera functions
    def start_webcam(self, port:int=None):
        """
        Starts the webcam and updates the camera image in the GUI.

        Parameters:
            port (int): The camera port to start (default is None).
        """
        # check for a available camera port
        if port == None:
            print("Warning: No camera detected!")
            return

        # Start the webcam
        self.video_capture = cv2.VideoCapture(port)

        # Create a Tkinter Label to display the camera image
        self.camera_label = ctk.CTkLabel(self, text="")
        self.camera_label.grid(row=0, column=1, rowspan=7, padx=20, pady=20)

        # Start the update function
        self.update_camera()

    def update_camera(self):
        """
        Continuously updates the camera image in the GUI.
        """
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
        """
        Stops the webcam and clears the camera image from the GUI.
        """
        self.after_cancel(self.update_camera)
        self.video_capture.release()
        self.camera_label.destroy()

    def get_connected_camera_ports(self):
        """
        Gets a list of connected camera ports.

        Returns:
            list: A list of available camera ports.
        """
        connected_ports = []

        # Start with index 0 and increment until no camera is found
        index = 0
        while True:
            # Try to open the camera with the current index
            try:
                cap = cv2.VideoCapture(index)
            except:
                pass
            
            # Check if the camera is opened successfully
            if not cap.isOpened():
                if not connected_ports: # add label if no camera is available
                    connected_ports.append("no cameras")
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