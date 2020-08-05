import os

import cv2

image_path = "sample"


def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    for i, face in enumerate(cascade.detectMultiScale(img)):
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]

        cv2.imwrite(os.path.join("faces", "{}_{}.jpg".format(imgname, i)), sub_face)

if __name__ == '__main__':

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Iterate through files
    for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:

        save_faces(cascade, f)