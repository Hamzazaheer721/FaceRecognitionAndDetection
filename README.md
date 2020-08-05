# FaceRecognitionAndDetection
This project will recognize the face from the trained datasets and then it will detect faces as who really the person is when you turn on your webcam and use video capturing option in OpenCV.

Make two folders named "sample" and "faces" in code folder.

1) For creating your own dataset, take your images and put them in your "sample" folder and run the face_crop.py
2) it will create optimized photos in "faces" folder.
3) now make label for the photos of the person whose dataset you are making first. 
   E.G in my case I created two directories in images folders with names of the person whose dataset I made
4) cut the photos from faces folder and paste them in your XYZ folder in images folder
5) repeat these processes for another labeled XYZ folder in images folder
6) Data will not be trained and will give error if you will train only one dataset of person e.g If you have only one folder in images folder.
7) after being done with creating dataset, Run face_train.py and train the data.
8) Run the face.py now and test your dataset and your results.

Note) Better the dataset you make, the better will be results.
Make sure you read the report and watch the video's ending if you can't understand hindi/urdu as last two minutes of videos shows my results.

Happy Coding!!!!
