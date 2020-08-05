import cv2 as cvE
import pickle

DATA = "./data/"
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {}
with open("label.pickle", 'rb') as f:
    oglabels = pickle.load(f) # we will get labels as hamza : 1 so we gotta make it 1: hamza so reversing it now
    labels = {v: k for k, v in oglabels.items()}
print(labels)

video = cv.VideoCapture(0)

while True:
    # capture frame by frame
    check, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(20, 20))
    for x, y, w, h in faces:
        print(x, y, w, h)
        target = gray[y:y + h, x:x + w]     # y_cord start - y_cord end
        targetC = frame[y:y + h, x: x + w]
        # prediction
        id_, conf = recognizer.predict(target)
        if conf >=45:
            print("confidence", conf)
            print(id_)   # fetched from data like if attique then 0 if hamza then 1
            print(labels[id_])   # our labels collected from pickle file like {attique : 0 hamza: 1} like labels[0] = hamza as our label is reversed now
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2  # line thickness
            cv.putText(frame, name, (x, y), font, 1, color, stroke, cv.LINE_AA)
        name = "face.jpg"
        cv.imwrite(name, target)  # this will make image of face in video capture per frame
        end_cord_x = x + w
        end_cord_y = y + h
        color = (255, 0, 0)
        stroke = 2
        cv.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv.destroyAllWindows()
