from flask import Flask,render_template, Response
import io
import cv2
import pickle
from flask_opencv_streamer.streamer import Streamer


app = Flask(__name__)

port = 3030
require_login = False
streamer = Streamer(port, require_login)

cap = cv2.VideoCapture(0)


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained.yml')

labels = dict() 

with open("label.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}


def gen():
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

        print("Found {0} faces!".format(len(faces)))
        count = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_frame = gray[y:y+h,x:x+w] 
            roi_gray = gray[y:y+h,x:x+w]
            id,conf = recognizer.predict(roi_gray)
            if conf >= 45 and conf <= 85:
                font = cv2.FONT_HERSHEY_SIMPLEX
                face_name = labels[id]
                cv2.putText(frame, face_name, (x,y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)