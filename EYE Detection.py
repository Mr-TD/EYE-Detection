import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
vid = cv2.VideoCapture(0)
while True:
    flag, fream = vid.read()
    gray = cv2.cvtColor(fream, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(fream, (x, y), (x + w, y + h), (255, 0, 0), 2)
        grayeye = gray[y:y + h, x:x + w]
        coloreye = fream[y:y + h, x:x + w]
        eyes = eye.detectMultiScale(grayeye)
        for (x2, y2, w2, h2) in eyes:
            cv2.rectangle(coloreye, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
    cv2.imshow('Result', fream)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
vid.release()
cv2.destroyAllWindows()


