import os
import cv2 as cv
import pygame
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import math
import cvzone

app = Flask(__name__, template_folder='templates')

# Load YOLO model
model = YOLO('fire.pt')

# Initialize PyGame for sound
pygame.init()
pygame.mixer.init()
alarm_sound = 'static/alert.mp3'
pygame.mixer.music.load(alarm_sound)

# Global variables to store detection state
view = None
fireDtc = False  # Track fire detection status

# Get the class ID for "fire" from model.names
fireId = None
for clsId, name in model.names.items():
    if name == 'fire':
        fireId = clsId
        break

# Camera setup
def camSetup():
    global view
    if view is None or not view.isOpened():
        view = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not view.isOpened():
            print("Error: Unable to access the camera")
            return False
    return True

# Image frame generator for video feed
def genFrames():
    global view, fireDtc
    frmCnt = 0
    frmSkp = 2
    while True:
        success, frm = view.read()
        if not success:
            print("Error: Failed to capture video frame")
            break
        else:
            frmCnt += 1
            if frmCnt % frmSkp != 0:
                continue

            # Run YOLO detection on the frame
            results = model(frm)

            fireDtc = False
            for result in results:
                for box in result.boxes:
                    if box.cls == fireId:
                        fireDtc = True
                        confidence = box.conf[0]
                        confidence = math.ceil(confidence * 100)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw the detection box around the fire
                        cv.rectangle(frm, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frm, f"Fire Detected {confidence}%", [x1 + 8, y1 + 30],
                                           scale=1.5, thickness=2, offset=10)
                        pygame.mixer.music.play()
                        break

            if not fireDtc:
                pygame.mixer.music.stop()

            # Encode the frame to JPEG format
            ret, buffer = cv.imencode(".jpg", frm)
            if not ret:
                print("Error: Frame encoding failed")
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Flask route for main page
@app.route("/")
def main():
    return render_template('index.html')

# Video feed route
@app.route("/video_feed")
def frames():
    if camSetup():
        return Response(genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Error: Camera not accessible"

# Fire status API
@app.route("/fire_status")
def fire_status():
    return jsonify(fireDtc=fireDtc)

# About and contact routes
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if view is not None:
            view.release()
