# Design refer to example on: https://github.com/TomSchimansky/CustomTkinter

import customtkinter as ctk
import os
import cv2
import re
from PIL import Image
from tkinter import filedialog

from frame_show_images import ImageFrame
import src.image_processing as imPr

import src.namespace as names

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
    def __init__(self, *args, **kwargs):
        # Set variables
        self.load_location_path: str = None
        self.save_location_path: str = None
        self.current_camera_port: str = None

        self.dataset = []

        self.ns = names.Names()

        # Initialize Tkinter
        ctk.CTk.__init__(self, *args, **kwargs)

        # Set window dimensions
        self.geometry(f"{1100}x{610}")

        # Set window title
        self.title('Analog Scanner')

        # Set window icon
        self.after(201, lambda: self.iconbitmap('imagesGUI\HHN_Logo_D_weiss.ico'))

        # Set GUI image paths
        gui_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "imagesGUI")
        self.logo_image = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "HHN_Logo_D_weiss.png")), size=(138, 58))
        self.play_icon = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "play_white_icon.jpg")), size=(20, 20))
        self.save_icon = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "save_white_icon.jpg")), size=(20, 20))
        self.load_icon = ctk.CTkImage(Image.open(os.path.join(gui_image_path, "folder_white_icon.jpg")), size=(20, 20))

        # Configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=8, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # Label with HHN logo
        self.navigation_frame_label = ctk.CTkLabel(self.sidebar_frame, text=None, image=self.logo_image,
                                                   compound="center", corner_radius=8)
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # OptionMenu for choosing between Image and Camera
        self.sidebar_camera_image = ctk.CTkOptionMenu(self.sidebar_frame, values=[self.ns.name_mode_image, self.ns.name_mode_camera],
                                                      command=self.sidebar_cam_img_event)
        self.sidebar_camera_image.grid(row=1, column=0, padx=20, pady=10)
        # Button to load image location
        self.sidebar_load = ctk.CTkButton(self.sidebar_frame, text="Load Location", image=self.load_icon, anchor='w',
                                          command=self.sidebar_btn_load_event)
        self.sidebar_load.grid(row=2, column=0, padx=20, pady=10)

        # OptionMenu for selecting the camera port
        self.sidebar_camera_port = ctk.CTkOptionMenu(self.sidebar_frame, values=self.get_connected_camera_ports(),
                                                      command=self.sidebar_cam_port_event)
        # self.sidebar_camera_port.grid(row=2, column=0, padx=20, pady=10)  # Show sidebar_load button first

        # OptionMenu for choosing image format
        self.sidebar_img_format = ctk.CTkOptionMenu(self.sidebar_frame, values=[self.ns.name_small_format, self.ns.name_medium_format, self.ns.name_dia],
                                                    command=self.sidebar_format_event)
        self.sidebar_img_format.grid(row=3, column=0, padx=20, pady=10)

        # OptionMenu for choosing image colortype
        self.sidebar_img_negativeType = ctk.CTkOptionMenu(self.sidebar_frame,
                                                          values=[self.ns.name_negative_bw, self.ns.name_negative_color,
                                                                  self.ns.name_positive],
                                                          command=self.sidebar_format_event)
        self.sidebar_img_negativeType.grid(row=4, column=0, padx=20, pady=10)

        # Checkbox to enable/disable image saving
        self.sidebar_save_cb = ctk.CTkCheckBox(self.sidebar_frame, text="Save Images", command=self.sidebar_cb_save_event)
        self.sidebar_save_cb.grid(row=5, column=0, padx=20, pady=10)

        # Button to set save location
        self.sidebar_save_location = ctk.CTkButton(self.sidebar_frame, text="Save Location", image=self.save_icon,
                                                    anchor='w', command=self.sidebar_btn_save_event)
        self.sidebar_save_location.grid(row=6, column=0, padx=20, pady=10)

        # Button to initiate image processing
        self.sidebar_process = ctk.CTkButton(self.sidebar_frame, text="Process", image=self.play_icon, anchor='w',
                                             command=self.sidebar_btn_process_event)
        self.sidebar_process.grid(row=7, column=0, padx=20, pady=10)

        # Label and OptionMenu for selecting appearance mode
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 10))

        # Label and OptionMenu for selecting UI scaling
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=11, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                     command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=12, column=0, padx=20, pady=(10, 20))


        # Set default values
        self.sidebar_camera_image.set(self.ns.name_mode_image)
        self.sidebar_img_format.set(self.ns.name_small_format)
        self.sidebar_img_negativeType.set(self.ns.name_negative_bw)
        self.sidebar_save_cb.deselect()
        self.sidebar_save_location.configure(state="disabled")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("90%")

        # Create a Tkinter Label to display the camera image
        self.camera_label = ctk.CTkLabel(self, text="")
        
        # Create image frame
        self.image_frame = ImageFrame(self)

        self.processing = imPr.ImageProcessing()

    #----------------------------------------------------------------------------------------------------
    # Sidebar callback functions
    def sidebar_cam_img_event(self, option:str):
        """
        Handles events when switching between Image and Camera modes.

        Parameters:
            option (str): The selected mode, either "Image" or "Camera".
        """
        if option == self.ns.name_mode_camera:
            self.sidebar_load.grid_forget()
            self.sidebar_camera_port.grid(row=2, column=0, padx=20, pady=10)

            # Remove image frame
            self.image_frame.grid_forget()

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
        # Reset dataset
        self.dataset = []

        self.load_location_path = filedialog.askopenfilename(initialdir='Images', title='Select a image!', multiple=True, defaultextension='.png', filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("GIF", "*.gif"), ("All Files", "*.*")])
        for img_path in self.load_location_path:
            img = cv2.imread(img_path)
            self.dataset.append(img)

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
        boundaryType = self.sidebar_img_format.get()
        negativeType = self.sidebar_img_negativeType.get()
        finished_imgs = []
        print(self.sidebar_camera_image.get())

        if self.sidebar_camera_image.get() == self.ns.name_mode_image:
            for img in self.dataset:
                ### Cut Images ###
                strips = self.processing.cutStrip(img, boundaryType=boundaryType, visualizeSteps=True)
                for strip in strips:
                    # self.processing.showImg(window_name='strip', img=strip)
                    height, width = img.shape[:2]
                    if height > width:
                        strip = cv2.rotate(src=strip, rotateCode=cv2.ROTATE_90_CLOCKWISE)

                    single_images, strip = self.processing.cutSingleImgs(strip, visualizeSteps=True, boundaryType=boundaryType)
                    print(f'[INFO] Found {len(single_images)} single images')

                    if self.sidebar_img_format.get != self.ns.name_dia:
                        finished_imgs = []
                        for img in single_images:
                            ### Invert images if needed ###
                            if strip is not None:
                                invertedImage = self.processing.invertImg(negative_img=img, offset_img=strip,
                                                               negative_type=negativeType, visualizeSteps=True)
                                finished_imgs.append(invertedImage)

                                #self.image_frame.update_images(finished_imgs)
        # Try to use camera
        else:
            try:
                # ToDo: get the actual image of the camera
                img = self.getCamImage()

                ### Cut Images ###
                strips = self.processing.cutStrip(img, boundaryType=boundaryType, visualizeSteps=True)
                for strip in strips:
                    # self.processing.showImg(window_name='strip', img=strip)
                    height, width = img.shape[:2]
                    if height > width:
                        strip = cv2.rotate(src=strip, rotateCode=cv2.ROTATE_90_CLOCKWISE)

                    single_images, strip = self.processing.cutSingleImgs(strip, visualizeSteps=True,
                                                                         boundaryType=boundaryType)
                    print(f'[INFO] Found {len(single_images)} single images')

                    if self.sidebar_img_format.get != self.ns.name_dia:
                        finished_imgs = []
                        for img in single_images:
                            ### Invert Images if needed ###
                            if strip is not None:
                                invertedImage = self.processing.invertImg(negative_img=img, offset_img=strip,
                                                                          negative_type=negativeType,
                                                                          visualizeSteps=True)
                                finished_imgs.append(invertedImage)
            except:
                print("[INFO] Not possible to load and process Image from Camera")
                pass

        #TODO: just for testing GUI
        if self.sidebar_camera_image.get() == self.ns.name_mode_image:
            pass
            #self.stop_webcam()
            # self.image_frame.grid(row=0, column=1, rowspan=3, columnspan=2, padx=10, pady=10)
            #self.image_frame.update_images(example_function_to_display_images())

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
        print(f'video_capture {self.video_capture}')

        # Set camera_label grid
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
            # Create a CTkImage object
            ctk_image = ctk.CTkImage(Image.fromarray(rgb_image), size=(850, 500))
            # Update the image in the Tkinter Label
            self.camera_label.configure(image=ctk_image)
            self.camera_label.image = ctk_image
        # Update the image again after a certain time (here: 30 milliseconds)
        self.after(30, self.update_camera)

    def getCamImage(self):
        """
        Captures a single frame from the video capture source (camera).

        Returns:
            numpy.ndarray or None: The captured frame in RGB format (using OpenCV).
                                  Returns None if no frame is captured.
        """
        _, frame = self.video_capture.read()  # Capture a frame from the video source

        if frame is not None:
            # Convert the frame from BGR to RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb_frame
        else:
            # Return None if no frame is captured
            return None

    def stop_webcam(self):
        """
        Stops the webcam and clears the camera image from the GUI.
        """
        self.after_cancel(self.update_camera)
        try:
            self.video_capture.release()
        except:
            print("[INFO] no camera frame on work")
        self.camera_label.grid_forget()

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
            print(f'index: {index}')
            # Try to open the camera with the current index
            print('[DEBUG] try connect camera')
            cap = cv2.VideoCapture(index)

            # Always release the camera capture object
            try:
                # Check if the camera is opened successfully
                if cap.isOpened():
                    print('[INFO] Camera found on port-' + str(index))
                    connected_ports.append("port-" + str(index))
                else:
                    print('[DEBUG] cap not open')
                    if not connected_ports and index == 0:  # add label if no camera is available
                        print('[INFO] No Camera found')
                        connected_ports.append("no cameras")
                    break
            finally:
                # Release the camera capture object
                print('[DEBUG] releases cap')
                cap.release()

            # Increment the index for the next iteration
            index += 1

            # Check for a condition to exit the loop (if needed)

        return connected_ports
        

# This function is to show example images with the image frame. It returns the images
def example_function_to_display_images():
    example_image_paths = ["Images/Analogscan043.jpg", "Images/Analogscan044.jpg", "Images/Analogscan045.jpg", "Images/Analogscan043.jpg", "Images/Analogscan044.jpg", "Images/Analogscan045.jpg"]
    imgs = []
    for path in example_image_paths:
        imgs.append(Image.open(path))

    return imgs

if __name__ == '__main__':
    app = App()
    app.mainloop()