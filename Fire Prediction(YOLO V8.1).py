from ultralytics import YOLO
import cvzone
import cv2
import math
import pygame

# Initialize pygame mixer for sound
pygame.mixer.init()
sirene_sound = pygame.mixer.Sound('static/alert.mp3')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
model = YOLO('fire.pt')

classnames = ['fire']
frame_count = 0
frame_skip = 2
fire_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame,(640, 480))
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence>50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 30],
                                   scale=1.5, thickness=2, offset=10)
                
                if not fire_detected:
                    sirene_sound.play(loops=-1)  # Play sound in a loop
                    fire_detected = True

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


