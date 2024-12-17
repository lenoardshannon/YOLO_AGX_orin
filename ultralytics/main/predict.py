from ultralytics import YOLO
import numpy as np
import cv2
# Load a COCO-pretrained YOLOv8n model
# model = YOLO("yolov8n.pt")
#
# # Display model information (optional)
# model.info()
#
# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=10, imgsz=640)
#
# # Run inference with the YOLOv8n model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")


model = YOLO('../../yolov8m.pt')

#results = model.track(source="../../football.mp4",show=True,tracker="bytetrack.yaml")
results = model.track(source=0,show=True,tracker="bytetrack.yaml")