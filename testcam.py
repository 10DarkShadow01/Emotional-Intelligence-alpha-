"""
Created on Fri Jul 13 17:48:57 2018

"""

import numpy as np
import cv2
from keras.preprocessing import image


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while(True):
    ret, img = cap.read()
    if ret is True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        continue

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		
        detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
        detected_face = cv2.resize(detected_face, (48, 48)) 
		
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
		
        img_pixels /= 255
		
        predictions = model.predict(img_pixels) 
		
        max_index = np.argmax(predictions[0])
		
        emotion = emotions[max_index]
		
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
cap.release()
cv2.destroyAllWindows()