import av
import logging
import streamlit as st
import queue
import cv2
import numpy as np
import threading

from PIL import Image
from consts import MODEL, COLORS, CLASSES, DEFAULT_CONFIDENCE_THRESHOLD, PROTOTXT, DEMO_IMAGE

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings
from typing import NamedTuple, List

# ----magic Function
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

logger = logging.getLogger(__name__)


def main():
    landing_page()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def landing_page():
    st.title("Detect any object")

    """Use the side bar to change the app mode"""

    live_object_detection_page = "Detect from camera"
    still_detection_page = "Upload an image to detect"

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            live_object_detection_page,
            still_detection_page,
        ],
    )
    st.subheader(app_mode)

    if app_mode == still_detection_page:
        still_object_detection()
    elif app_mode == live_object_detection_page:
        live_object_detection()


def live_object_detection():
    st.header("Camera feed detection")

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                PROTOTXT, MODEL
            )
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    name = CLASSES[idx]
                    result.append(Detection(name=name, prob=float(confidence)))

                    # display the prediction
                    label = f"{name}: {round(confidence * 100, 2)}%"
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, result = self._annotate_image(image, detections)

            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=MobileNetSSDVideoProcessor,
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()

            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break


def still_object_detection():

    @st.cache
    def process_image(image):
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        net.setInput(blob)
        detections = net.forward()
        return detections

    @st.cache
    def annotate_image(
            image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
    ):
        # loop over the detections
        (h, w) = image.shape[:2]
        labels = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
                labels.append(label)
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
                )
        return image, labels

    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        """This is the default image, upload yours to see detections."""
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    detections = process_image(image)
    image, labels = annotate_image(image, detections, confidence_threshold)

    st.image(
        image, caption=f"Processed image", use_column_width=True,
    )

    st.write(labels)


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
               "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
