
 # This script will detect faces via your webcam or any video
# Tested with OpenCV3

import cv2
import pickle
from flask_opencv_streamer.streamer import Streamer

#cap = cv2.VideoCapture("v.mp4") # path of video or put cv2.VideoCapture(0) for webcam

port = 3030
require_login = False
streamer = Streamer(port, require_login)

cap = cv2.VideoCapture(0)


# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained.yml')

labels = dict() 

with open("label.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		
	)

	print("Found {0} faces!".format(len(faces)))
	count = 0
	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		roi_frame = gray[y:y+h,x:x+w] 
		roi_gray = gray[y:y+h,x:x+w]
		id,conf = recognizer.predict(roi_gray)
		if conf >= 45 and conf <= 85:
		    #print(id_)
		    #print(labels[id_])
		    font = cv2.FONT_HERSHEY_SIMPLEX
		    name = labels[id]
		    color = (255, 255, 255)
		    stroke = 2
		    cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		#cv2.imwrite(frame,roi_frame)
		
		cv2.imwrite("test_faces/teeest{}.png".format(count),roi_gray)
		count += 1

	# Display the resulting frame
	streamer.update_frame(frame)
	if not streamer.is_streaming:
	    streamer.start_streaming()
	
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
