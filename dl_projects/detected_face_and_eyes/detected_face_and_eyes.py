import cv2
import numpy as np

# Добавить полный путь к файлу с каскадами
haar_cascade_face = cv2.CascadeClassifier("path/haarcascade_frontalface_default.xml")
haar_cascade_eye = cv2.CascadeClassifier("path/haarcascade_eye.xml")

cap = cv2.VideoCapture(4)

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    find_face = haar_cascade_face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in find_face:
        cv2.putText(img, 'face', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        gray_coord = gray[y:y + h, x:x + w]
        color_coord = img[y:y + h, x:x + w]

        find_eyes = haar_cascade_eye.detectMultiScale(gray_coord)

        for (ex, ey, ew, eh) in find_eyes:
            cv2.putText(color_coord, 'eyes', (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(color_coord, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    cv2.imshow('Face & Eye Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
