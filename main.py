import cv2
import numpy as np
import winsound
import pandas as pd
import streamlit as st
from datetime import datetime

# Load YOLO
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Create a DataFrame to store object name and time data
df = pd.DataFrame(columns=['Object Name', 'Start Time', 'End Time'])

# Function to record object name and time
def record_object_time(object_name, start_time, end_time):
    global df
    df = pd.concat([df, pd.DataFrame({'Object Name': [object_name], 'Start Time': [start_time], 'End Time': [end_time]})], ignore_index=True)

# Function to run the object detection and recording
def run_object_detection():
    # Open webcamera
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to the appropriate index if you have multiple webcams

    alarm_active = False
    alarm_duration = 500  
    alarm_frequency = 1500  

    while True:
        ret, frame = cap.read()

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo.setInput(blob)
        outputs = yolo.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Check if a person is detected
        alarm_active = any(classes[class_ids[i]] == 'person' for i in indexes)

        # Record object name and time if a person is detected
        if alarm_active:
            current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
            if not alarm_active:
                record_object_time('person', current_time, None)
            else:
                record_object_time('person', None, current_time)

            # Play the alarm sound
            winsound.Beep(alarm_frequency, alarm_duration)

        # Draw rectangles and display labels
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), colorGreen, 3)

                # Display class label and confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, 2)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the recorded data to an Excel file
    df.to_excel('D:/mm/object_time_records.xlsx', index=False)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
def main():
    st.title("Object Detection and Recording")

    # Display webcam feed
    st.markdown("**Webcam Feed**")
    st.image(run_object_detection(), channels="BGR")

if __name__ == "__main__":
    main()
