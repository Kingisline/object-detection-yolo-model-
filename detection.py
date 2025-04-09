import cv2
import time
import random
import argparse
import numpy as np
import onnxruntime as ort

# Function to load the source (image, video, or webcam)
def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # List of image formats
    key = 1 # 1 = Video, 0 = Image
    frame = None
    cap = None

    # If source is a webcam
    if(source_file == "0"):
        image_type = False  # Not an image, it's video from the webcam
        source_file = 0    
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats  # Check if source is an image

    # Open image or video source
    if(image_type):
        frame = cv2.imread(source_file)  # Read image
        key = 0  # Set key for image
    else:
        # cap = cv2.VideoCapture(source_file)  # Open video capture for video or webcam
        cap = cv2.VideoCapture(0)

    return image_type, key, frame, cap

if __name__ == '__main__':
    # Add argument parser for command line input
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")  # Video source
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")  # File containing class names
    parser.add_argument("--model", type=str, default="yolo11n.onnx", help="Pretrained Model")  # ONNX model file
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")  # Confidence threshold for detection
    parser.add_argument("--thickness", type=int, default=2, help="Line Thickness on Bounding Boxes")  # Thickness of bounding box lines
    args = parser.parse_args()    

    # Load the YOLO model
    model = cv2.dnn.readNet(args.model)

    IMAGE_SIZE = 640  # Set the image size for the model
    NAMES = []
    # Read the object names from the specified file
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]  # Generate random colors for each object class

    source_file = args.source    
    # Load the source (image, video, or webcam)
    image_type, key, frame, cap = loadSource(source_file)
    grabbed = True

    while(1):
        # For video input, read the next frame
        if not image_type:
            (grabbed, frame) = cap.read()

        # Exit if no more frames are grabbed
        if not grabbed:
            exit()

        # Make a copy of the frame
        image = frame.copy()
    
        # Prepare image as input for the YOLO model
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)

        # Initialize lists for detection results
        class_ids, confs, boxes = list(), list(), list()

        # Set input to the model and perform forward pass
        model.setInput(blob)
        preds = model.forward()
        preds = preds.transpose((0, 2, 1))  # Adjust output shape

        # Calculate scaling factors based on image size
        image_height, image_width, _ = image.shape
        x_factor = image_width / IMAGE_SIZE
        y_factor = image_height / IMAGE_SIZE

        rows = preds[0].shape[0]

        # Iterate over each prediction row
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]  # Confidence score
            
            # Extract class scores and find the class with the highest score
            classes_score = row[4:]
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if (classes_score[class_id] > .25):  # Filter out weak predictions
                confs.append(classes_score[class_id])  # Store confidence
                label = NAMES[int(class_id)]  # Get class label
                class_ids.append(class_id)  # Store class ID
                
                # Extract bounding box coordinates
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)  # Store bounding box coordinates

        # Apply Non-Maximum Suppression (NMS) to eliminate overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)         

        # Draw bounding boxes and labels on the image
        for i in indexes:
            box = boxes[i]
            class_id = class_ids[i]
            score = confs[i]

            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # Draw bounding box on the image
            cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[class_id], args.thickness)
            
            # Prepare label with class name and confidence score
            name = NAMES[class_id]    
            score = round(float(score), 3)  # Round confidence score            
            name += f' {str(score)}'

            # Add text label above the bounding box
            font_size = args.thickness / 2.5
            margin = args.thickness * 2
            cv2.putText(image, name, (left, top - margin), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[class_id], args.thickness)

        # Indicate the image has been processed
        grabbed = False
        # Display the processed image with detections
        cv2.imshow("Detected",image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(key) ==  ord('q'):
            break