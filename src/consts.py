import numpy as np

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODEL = "C://Users//natha/PycharmProjects//object_detection/models/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "C://Users//natha/PycharmProjects//object_detection/models/model_text.prototxt"

DEMO_IMAGE = "C://Users//natha/PycharmProjects//object_detection//images/demo.jpg"
