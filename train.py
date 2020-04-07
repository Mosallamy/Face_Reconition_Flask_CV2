import os
from PIL import Image
import numpy as np
import cv2
import pickle 
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


DIR =os.path.dirname(os.path.abspath(__file__))
faces = os.path.join(DIR,'images')

y_labels = []
x_train = []
label_ids = {}
current_id = 0

for root, dirs, files in os.walk(faces):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label, " " ,path )

            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            #print(label_ids)
            
            image = Image.open(path).convert("L")
            image = image.resize((550,550),Image.ANTIALIAS)
            image_array = np.array(image,"uint8")
            #print(image_array)
            faces = faceCascade.detectMultiScale(
                image_array,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


print(y_labels)
print(x_train)

with open("label.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train, np.array(y_labels))

recognizer.save("trained.yml")
