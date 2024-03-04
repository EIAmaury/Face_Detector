import cv2

#pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#detect face
img=cv2.imread('Family.jpg')
#grayscale conversion
grayscaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#detect face and returns the top letf corner coordinates and the width and height of the rectangle
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
#draw rectangle around the face
for(x,y,z,w) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+z,y+w),(0,255,0),2)
    cv2.circle(img,(x+z//2,y+w//2),w//2,(0,0,255),2)
#display image
cv2.imshow('RDJ',grayscaled_img)
cv2.imshow('RDJ',img)
#wait key to see the image
#press one key to stop the image
cv2.waitKey()
print("Code completed")