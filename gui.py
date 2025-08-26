import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import winsound  

class ObjectDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize YOLO
        self.yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as file:
            self.classes = [line.strip() for line in file.readlines()]
        layer_names = self.yolo.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]

        # Initialize variables
        self.image_path = ""
        self.video_source = 0  # Default webcam index
        self.vid = cv2.VideoCapture(self.video_source)
        self.canvas = tk.Canvas(window)
        self.canvas.pack()

        # Open Image button
        self.btn_open_image = tk.Button(window, text="Open Image", width=15, command=self.open_image)
        self.btn_open_image.pack()

        # Open Webcam button
        self.btn_open_webcam = tk.Button(window, text="Open Webcam", width=15, command=self.open_webcam)
        self.btn_open_webcam.pack()

        # Detect Objects button
        self.btn_detect_objects = tk.Button(window, text="Detect Objects", width=15, command=self.detect_objects)
        self.btn_detect_objects.pack()

        # Image variable for displaying in the canvas
        self.image_var = None

        # Flag to indicate if the webcam is open
        self.webcam_open = False
        self.lock = threading.Lock()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # If the webcamera is open, release it
            self.release_webcam()

            self.image_path = file_path
            self.update_canvas()

    def open_webcam(self):
        # If an image is currently loaded, clear it
        self.image_path = ""
        self.update_canvas()

        # If the webcam is already open, do nothing
        if self.webcam_open:
            return

        # Open the webcam in a separate thread
        self.webcam_thread = threading.Thread(target=self.open_webcam_thread)
        self.webcam_thread.start()

    def open_webcam_thread(self):
        self.vid = cv2.VideoCapture(self.video_source)
        self.webcam_open = True
        self.update_canvas()

        while self.webcam_open:
            ret, frame = self.vid.read()
            if ret:
                try:
                    frame = self.detect_objects_in_image(frame)
                    self.update_canvas(frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    break
            else:
                print("Error reading frame.")
                # Release the webcam if an error occurs
                self.release_webcam()
                break

        # Release the webcam when the thread exits
        self.vid.release()

    def detect_objects(self):
        if self.image_path:
            # If an image is loaded, perform detection on the image
            frame = cv2.imread(self.image_path)
            frame = self.detect_objects_in_image(frame)

            # Update canvas with the annotated image
            self.update_canvas(frame)
        elif self.webcam_open:
            # If the webcam is open, do nothing here
            pass

    def detect_objects_in_image(self, frame):
        # YOLO object detection code (same as before)
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo.setInput(blob)

        outputs = self.yolo.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Initialize alarm parameters
        alarm_active = False
        alarm_duration = 500  # in milliseconds
        alarm_frequency = 1500  # in Hertz

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Display class label and confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check if the detected object is of interest (e.g., person)
                if label == 'person':
                    # Activate the alarm
                    alarm_active = True

        # Check if the alarm should be active
        if alarm_active:
            # Play the alarm sound
            winsound.Beep(alarm_frequency, alarm_duration)

        return frame

    def update_canvas(self, frame=None):
        if frame is None:
            if self.image_path:
                # If an image is loaded, display it on the canvas
                frame = cv2.imread(self.image_path)
            elif self.webcam_open:
                # If the webcam is open, capture a frame
                ret, frame = self.vid.read()
                if not ret:
                    return

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to ImageTk format
        image_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

        # Update the canvas with the new image
        with self.lock:
            self.canvas.config(width=image_tk.width(), height=image_tk.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

            # Save the reference to the ImageTk object to prevent it from being garbage collected
            self.image_var = image_tk

    def release_webcam(self):
        # Release the webcam and set the flag to False
        self.webcam_open = False
        if hasattr(self, 'webcam_thread'):
            self.webcam_thread.join()

# Create a window and pass it to the ObjectDetectionApp class
root = tk.Tk()
app = ObjectDetectionApp(root, "Object Detection App")
root.mainloop()
