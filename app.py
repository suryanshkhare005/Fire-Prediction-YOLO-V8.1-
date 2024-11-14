import os
import cv2
import pygame
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__, template_folder='templates')



# Load YOLO model
model = YOLO('fire.pt')

# Initialize PyGame for sound
pygame.init()
pygame.mixer.init()
alarm_sound = 'static/alert.mp3'
pygame.mixer.music.load(alarm_sound)

# Global flag to control detection state
detection_active = False

def generate_frames():
    global detection_active
    cap = cv2.VideoCapture(0)  # Open the default camera (0 for the first camera)

    while True:
        if detection_active:
            success, frame = cap.read()  # Capture a frame
            if not success:
                break  # If frame capture failed, exit the loop

            # Run YOLO detection on the frame
            results = model(frame)
            detections = results.pandas().xyxy[0]

            fire_detected = any(detections['name'] == 'fire')

            # Add a visual indicator if fire is detected
            if fire_detected:
                pygame.mixer.music.play()
                cv2.putText(frame, "Fire Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                pygame.mixer.music.stop()

            # Convert the frame to JPEG and yield it to the browser
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/start-detection', methods=['GET'])
def start_detection():
    global detection_active
    detection_active = True  # Start detection
    return jsonify({'message': 'Detection started'})

if __name__ == '__main__':
    app.run(debug=True)
