import os, cv2 as cv
from PIL import Image
import numpy as np
import pickle

y_labels = []
x_train = []
label_ids = {}
current_id = 0

Base_path = os.getcwd()     # tells the path where our running program is running but wont mention running folder name

img_dir = os.path.join(Base_path, "Images")

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(img_dir):
    # print(root)
    # print(dirs)
    # print(files)
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            # print(path)
            pathInwhichfile_exists = os.path.dirname(path)
            # print(pathInwhichfile_exists)
            label = os.path.basename(pathInwhichfile_exists).replace(" ", "-").lower()
            # print(label, pathInwhichfile_exists)
            # print("label :", label)
            if not label in label_ids:
                label_ids[label] = current_id
                # print(label_ids)
                # print(label_ids[label])
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            pil_img = Image.open(path).convert('L')     # converting it into grey scaled image
            size = (500, 500)
            final_img = pil_img.resize(size, Image.ANTIALIAS)
            img_array = np.array(final_img, "uint8")
            # print(img_array)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)
            for x, y, w, h in faces:
                roi = img_array[y: y+h, x: x+w]
                x_train.append(roi)
                y_labels.append(id_)
print(label_ids)   #{attique : 0 hamza : 1}
print(y_labels)  # [0000000111111]
# for storing all the data in the files and doing training part next step after this will be training
# in training you will have to read yml file
with open("label.pickle", 'wb') as f:     # The wb indicates that the file is opened for writing in binary mode.
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.write("trainer.yml")