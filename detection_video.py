# People Counter
# Ammar Chalifah (July 2020)

# Libraries
import numpy as np
import cv2
print(cv2.__version__)
import imutils
print(imutils.__version__)
import math
import os

from functions.centroidtracker import CentroidTracker
from functions.trackableobject import TrackableObject
from functions import config_util
from functions import label_map_util

from object_detection.builders import model_builder

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model

tf.get_logger().setLevel('ERROR')

# -----------
# HELPER CODE
# -----------

def clean_detection_result(boxes, classes, scores, classes_to_detect='person', threshold = 0.5):
    """
    Function to remove all prediction results that are not included in classes_to_detect.
    In default, this function will remove all predictions other than 'person' type.

    Args:
        boxes, classes, scores -> detection from TF2 Object Detection Model.
        classes_to_detect -> list of strings. default: ['person']
        threshold -> minimum score
    Returns:
        new_boxes, new_classes, new_scores -> numpy array of cleaned prediction result.
    """
    new_boxes = []
    new_classes = []
    new_scores = []
    for i in range(scores.shape[0]):
        if scores[i]>threshold and category_index[classes[i]]['name'] in classes_to_detect:
            new_boxes.append(boxes[i])
            new_classes.append(classes[i])
            new_scores.append(scores[i])
    return np.asarray(new_boxes), np.asarray(new_classes), np.asarray(new_scores)

#------------VIDEO STREAM--------------
# Define the video stream
cap = cv2.VideoCapture('video.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

#Target size of the video stream
font = cv2.FONT_HERSHEY_SIMPLEX

W = None
H = None

MODEL_NAME = 'efficientdet_d1_coco17_tpu-32'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'checkpoint/')
# List of the strings that is used to add correct label for each box.
PATH_TO_CFG = os.path.join(MODEL_NAME, 'pipeline.config')
PATH_TO_LABELS = os.path.join('label', 'mscoco_label_map.pbtxt')
# Number of classes to detect
NUM_CLASSES = 90

#----------------HUMAN COUNTER-----------------
humancounter = 0
first_detection = True

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#Object Tracking Helper Code
ct = CentroidTracker(maxDisappeared=30, maxDistance=150)
trackableObjects = {}

ppl = 0

# Model Loading
print('[INFO] loading detection model ...')
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training = False)

ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
print('[INFO] detection model loaded')

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

# Detection
while True:
    # Read frame from camera
    ret, image_np = cap.read()

    image_np = imutils.resize(image_np, width = 800)

    if W is None or H is None:
        (H, W) = image_np.shape[:2]

    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    rects = []

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1

    boxes = detections['detection_boxes'][0].numpy()
    classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    scores = detections['detection_scores'][0].numpy()

    # Remove all results that are not a member of [classes_to_detect] and has score lower than 50%
    boxes, classes, scores = clean_detection_result(boxes, classes, scores, 'person',threshold = 0.5)

    # Bounding boxes
    for box in boxes:
        box = box*np.array([H, W, H, W])
        ymin, xmin, ymax, xmax = box.astype('int')

        # draw bounding boxes
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        rects.append((xmin, ymin, xmax, ymax))
    
    # use the centroid tracker to associate the old object centroids
    # with the newly computer object centroids
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        
        # check to see if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - y[0]
            to.centroids.append(centroid)

            if not to.counted:
                ppl += 1
                to.counted = True
        
        trackableObjects[objectID] = to

    cv2.putText(image_np, "people: {}".format(ppl), (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Display output
    cv2.imshow('object detection', image_np)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()