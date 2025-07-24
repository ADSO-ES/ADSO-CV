import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from ultralytics import YOLO

def run_detection(image_path, yolo_path, cnn_path, input_size=(64, 64)):
    # Load models
    yolo_model = YOLO(yolo_path)
    cnn_model = tf.keras.models.load_model(cnn_path)

    # Read image
    img = cv2.imread(image_path)
    results = yolo_model(img)

    combined_results = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]

        # Crop and resize
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, input_size)
        input_tensor = np.expand_dims(resized, axis=0)

        # CNN prediction
        cnn_pred = cnn_model.predict(input_tensor, verbose=0)[0]
        cnn_class = int(np.argmax(cnn_pred))
        cnn_conf = float(np.max(cnn_pred))

        warning = int(class_id != cnn_class)

        combined_results.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "yolo_class_id": class_id,
            "yolo_class_name": class_name,
            "yolo_confidence": conf,
            "cnn_predicted_class": cnn_class,
            "cnn_confidence": cnn_conf,
            "warning": warning
        })

    return pd.DataFrame(combined_results)
