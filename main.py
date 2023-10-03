import threading
import os
import cv2
from deepface import DeepFace

# 0 para selecionar o index das cameras disponíveis.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frames_counter = 0

face_match = False
face_name = ''

imgs_path = 'imgs/'
references = []

for file_path in os.listdir(imgs_path):
    if os.path.isfile(os.path.join(imgs_path, file_path)) and '.jpg' in file_path:
        img = cv2.imread(f"./{os.path.join(imgs_path, file_path)}")
        label = file_path.split('.')[0]
        references.append((img, label))


def check_face_match(frame):
    global face_match
    global face_name
    try:
        for img, label in references:
            if DeepFace.verify(frame, img.copy())['verified']:
                face_match = True
                face_name = label
                break
            else:
                face_name = ''
                face_match = False

    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if frames_counter % 30 == 0:
            try:
                threading.Thread(target=check_face_match, args=(frame.copy(),)).start()
            except ValueError:
                pass

        frames_counter += 1

        if face_match:
            cv2.putText(frame, f"MATCH! {face_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, f"NO MATCH! {face_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow("Cam", frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

