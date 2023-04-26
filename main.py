import cv2
from flask import Flask, render_template, Response
import numpy as np
import imutils
import os
import requests
from PIL import Image

app = Flask(__name__)

camera_url = 'http://192.168.3.130:8080/video'  # Replace with your IP camera URL

# Load the MobileNet-SSD model
prototxt = os.path.join(os.path.dirname(__file__), 'deploy.prototxt.txt')
caffemodel = os.path.join(os.path.dirname(__file__), 'mobilenet_iter_73000.caffemodel')
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# Define classes that the model can recognize
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

count = 1


def detect_dog(frame, conf_threshold=0.2):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    dog_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "dog":
                dog_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return dog_detected, frame


def detect_motion(frame, prev_frame, min_area=500):
    sensitivity_threshold = 25  # This is used to determine how sensitive motion detection should be.
    if prev_frame is None:
        return False, frame

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    frame_delta = cv2.absdiff(prev_frame_gray, frame_gray)
    thresh = cv2.threshold(frame_delta, sensitivity_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    motion_detected = False
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    return motion_detected, frame


def generate_frames():
    global count
    # resp = requests.get(camera_url, stream=True).raw
    # img = np.asarray(bytearray(resp.read()), dtype="uint8")
    # img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    # cap = cv2.VideoCapture(camera_url)
    prev_frame = None

    while True:
        cap = cv2.VideoCapture(camera_url)
        # resp = requests.get(camera_url, stream=True).raw
        # img = np.asarray(bytearray(resp.read()), dtype="uint8")
        # img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        ret, frame = cap.read()
        if not ret:
            break
        # if count == 1:
        #     dog_detected, frame = detect_dog(frame)
        # # movement, frame = detect_motion(frame, prev_frame)
        # # if movement:
        # #     print("There is movement.")
        # dog_detected, frame = detect_dog(frame)
        # if dog_detected:
        #     print("Dog detected!")
        #     if count == 1:
        #         img = Image.fromarray(frame)
        #         img.show()
        #         count -= 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
