import cv2
import time
import random
import argparse
import numpy as np
import onnxruntime as ort

def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    key = 1
    frame = None
    cap = None

    if source_file == "0":
        image_type = False
        source_file = "data/videos/road.mp4"  # or use 0 for webcam
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats

    if image_type:
        frame = cv2.imread(source_file)
        key = 0
    else:
        cap = cv2.VideoCapture(source_file)

    return image_type, key, frame, cap

def run_inference(session, input_name, blob):
    outputs = session.run(None, {input_name: blob})
    return outputs[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--model", type=str, default="yolo11n.onnx", help="Pretrained ONNX Model")
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Line Thickness on Bounding Boxes")
    args = parser.parse_args()

    # Load model with GPU and verify provider
    print("[INFO] Loading model with CUDAExecutionProvider...")
    session = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name

    providers = session.get_providers()
    print(f"[INFO] Current ONNX Runtime Execution Provider: {providers[0]}")
    if 'CUDAExecutionProvider' in providers:
        print("[INFO] ✅ Running on GPU (NVIDIA CUDA)")
    else:
        print("[WARNING] ⚠️ Not using GPU!")

    IMAGE_SIZE = 640
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()]
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    image_type, key, frame, cap = loadSource(args.source)
    grabbed = True

    prev_time = 0  # for FPS

    while True:
        if not image_type:
            (grabbed, frame) = cap.read()
        if not grabbed:
            break

        start_time = time.time()

        image = frame.copy()
        img_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)

        preds = run_inference(session, input_name, blob)
        preds = np.transpose(preds, (0, 2, 1))

        image_height, image_width, _ = image.shape
        x_factor = image_width / IMAGE_SIZE
        y_factor = image_height / IMAGE_SIZE

        class_ids, confs, boxes = [], [], []

        rows = preds[0].shape[0]
        for i in range(rows):
            row = preds[0][i]
            conf = row[4]
            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if classes_score[class_id] > args.tresh:
                confs.append(float(classes_score[class_id]))
                class_ids.append(class_id)
                x, y, w, h = row[0], row[1], row[2], row[3]
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])

        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)

        for i in indexes:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            box = boxes[i]
            class_id = class_ids[i]
            score = confs[i]
            left, top, width, height = box

            cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[class_id], args.thickness)
            label = f"{NAMES[class_id]} {round(score, 3)}"
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        args.thickness / 2.5, COLORS[class_id], args.thickness)

        # === FPS Display ===
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(image, f"FPS: {fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Detected (GPU)", image)
        if cv2.waitKey(key) == ord('q'):
            break
