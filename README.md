# Analog Scanner
Project in the lecture Computer- & Robotervision at  the Hochschule Heilbronn in the Master's program "Mechatronik and Robotik" at the Faculty of Technik ([TE](https://www.hs-heilbronn.de/de/fakultaet-te)). 

This project was supervised by Prof. Dr. rer. nat. Dieter Maier.

## Concept
Analog strips are photographed on an illuminated background. The rest is managed by digital image processing. 
The result is digitized images of the objects in the camera's field of view. 
Images can be loaded from Path or captured directly by a connected camera.

### Supported Film-Formats
| Format             | 35 mm | 120 (6x6)                                   | 120 (6x9) | Dias |
| :---: | :---: | :---: | :---: | :---: |
| Black &<br/>Whitze |![35mmSW.jpg](Images%2F35mmSW.jpg)| ![6x6SW.jpg](Images%2F6x6SW.jpg)            |Yes, No Example Image|Yes, No Example Image|
| Colored            |![35mmColor2.jpg](Images%2F35mmColor2.jpg)| Yes, No Example Image |![6x9Color.jpg](Images%2F6x9Color.jpg)|![HighRes_Dias.jpg](Images%2FHighRes_Dias.jpg)|


## Graphical User Interface (GUI)
The GUI forms the interface between the user and the system. 
It is designed to enable intuitive and efficient interaction with the application. 
The GUI simplifies the digitization of analog images and slides and creates a user-friendly environment.

![GUI.png](Images%2FREADME%20Images%2FGUI.png)

## Camera Support
The connection of a camera and the resulting live view of the images is also implemented.
When selecting the camera, all available camera ports are detected and listed in a drop-down menu. 
The first port is always selected automatically and its live image is included in the GUI preview.

## Project Goal
The goal of this project is to develop a fully functional and user-friendly program capable of digitizing analog photos and slides. 
The focus is on the following features:

1. Diverse digitalization options:<br>
The program aims to provide versatile digitization capabilities, allowing users to digitize pre-existing images from analog film strips and slides, as well as capture live images through a camera. 
This flexibility enables users to adapt the program to their specific needs.

2. Precise Image Recognition and Cropping:<br>
After specifying standardized formats, the program will automatically recognize, crop, and optimally align images/strips. 
This ensures efficient digitization with minimal manual intervention, streamlining the overall process.

3. Inversion and Color Correction: <br>
Leveraging modern image processing techniques, the program will invert images and apply color correction to enhance their quality, bringing them as close to the original as possible.

4. User-Friendly Interface:<br>
The user interface of the program will be designed to be intuitive, catering to both technically proficient users and hobby photographers. 
The goal is to provide a straightforward application experience, promoting ease of use.


## Projekt Setup
To use this project, you need to clone it from the GitHub repository. 
Follow the steps below in the command window (CMD) to navigate to the desired location where the project should be located:

#### Bash
```diff
cd Path\to\destination\directory
```
Clone the repository into the specified directory using the following command:
```diff
git clone https://github.com/Tim-Keicher/AnalogScannerCRV.git
```
Clone the repository into the specified directory using the following command:

## Environment Setup
As this project is programmed in Python, we recommend using an environment management tool like Anaconda. 
Follow the steps below to set up the environment:

1. Download and install Anaconda from https://www.anaconda.com/download.
2. Open the Anaconda Prompt.
3. Navigate to the cloned project directory from GitHub:
```diff
cd Path\to\destination\directory\AnalogScannerCRV
```
4. Generate the environment variable using the provided environment.yml file:
```diff
conda env create -f analogScan.yml
```

### Verify Environment Setup
To check if the environment variable was created successfully, run the following command:
```diff
conda env list
```

### Activate Environment
To activate the environment variable, run the following command:
```diff
conda activate analogScan
```

### Run the Scanner
The scanner program can now be launched using the following command:
```diff
python main.py
```

