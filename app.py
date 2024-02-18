import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch

# Initialize the GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)

def detect_and_draw_objects(frame):
    scale_percent = 50  # Adjust based on your requirement
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Convert the color space from BGR to RGB and create a PIL Image
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Process the image for the model
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Convert outputs to COCO API
    results = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]).to(device), threshold=0.9)[0]

    # Draw rectangles around detected objects
    for score, label, box in zip(results["scores"].cpu(), results["labels"].cpu(), results["boxes"].cpu()):
        box = [int(b * (100 / scale_percent)) for b in box.tolist()]  # Adjust based on resizing
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}",
                    (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def update_image():
    ret, frame = cap.read()
    if ret:
        # Process frame for object detection and drawing
        frame_with_detections = detect_and_draw_objects(frame)

        # Convert the frame to a format compatible with Tkinter
        frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=im)
        lbl.configure(image=img)
        lbl.image = img
    lbl.after(10, update_image)

# Initialize the main application window
root = tk.Tk()
root.title("Camera Feed")

# Create a label to display the camera frames
lbl = ttk.Label(root)
lbl.pack()

# Start the camera
cap = cv2.VideoCapture(0)

# Start the function to update images
update_image()

# Run the GUI loop
root.mainloop()

# Cleanup on close
cap.release()
cv2.destroyAllWindows()
