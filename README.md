# Real-Time Object Detection in Camera Feed

This application integrates a real-time camera feed with object detection capabilities using the DETR (DEtection
TRansformer) model with a ResNet-50 backbone. It displays the camera feed in a GUI window and draws bounding boxes
around detected objects, labeling them with their detected class and confidence score.

## Features

- Real-time object detection in camera feed
- Bounding box and label overlay on detected objects
- Utilizes GPU for improved performance (if available)

## Installation

Ensure you have Python 3.8 or later installed on your system. This application also requires pip for managing Python
packages.

### Setting Up Your Environment

1. Clone or download this repository to get a copy of the source code.

2. Navigate to the project directory where the requirements.txt is located.

3. Install the required dependencies by running the following command:

```sh
pip install -r requirements.txt
```

This command installs all the necessary Python packages listed in requirements.txt.

## Usage

To run the application, execute the script with Python from the terminal or command prompt:

```sh
python path/to/app.py
```

Replace path/to/app.py with the actual path to the script if necessary.

## How It Works

The script initializes a camera feed using OpenCV and processes each frame through the DETR model to detect objects.
Detected objects are highlighted with bounding boxes, and their labels are displayed on the GUI created with Tkinter.

## Customization

You can adjust the detection threshold and frame resizing parameters within the detect_and_draw_objects function to tune
the application's performance and detection sensitivity according to your needs.


