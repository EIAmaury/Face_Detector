import cv2

#pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector=cv2.CascadeClassifier('haarcascade_eye.xml')
#detect face from webcam
camera=cv2.VideoCapture(0)
while True:
    sucess,frame=camera.read()
    #grayscale conversion
    img_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect face and returns the top letf corner coordinates and the width and height of the rectangle
    face_coordinates=trained_face_data.detectMultiScale(img_grayscale)
    eye_coordinates=eye_detector.detectMultiScale(img_grayscale)
    #draw rectangle around the face
    for(x,y,z,w) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+z,y+w),(0,255,0),2)
        cv2.putText(frame,"Face",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),1,cv2.LINE_AA)
        
    for(x,y,z,w) in eye_coordinates:
        cv2.circle(frame,(x+z//2,y+w//2),w//2,(0,0,255),2)
    #display image
    cv2.imshow('face_detector',frame)
    key=cv2.waitKey(1)
    if key==113 or key==81 or key==ord('q'):
        break
camera.release()
