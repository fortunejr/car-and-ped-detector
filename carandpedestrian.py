import cv2

classifier_file_car = 'cardetector.xml'
classifier_file_ped = 'haarcascade_fullbody.xml'
# Our Image
img_file = 'C:\\Users\\user\\Pictures\\carimg.PNG'
# video = cv2.VideoCapture('dashcam3.MOV')
video = cv2.VideoCapture('car_and_pede.mp4')

car_traker = cv2.CascadeClassifier(classifier_file_car)
ped_tracker = cv2.CascadeClassifier(classifier_file_ped)
while True:
    read_successful, frame = video.read()
    
    if read_successful:
        grey_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    cars = car_traker.detectMultiScale(grey_scaled_frame)
    peds = ped_tracker.detectMultiScale(grey_scaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(grey_scaled_frame, (x, y), (x+w, y+h), (0, 0, 255),2)
    for (x,y,w,h) in peds:
        cv2.rectangle(grey_scaled_frame, (x, y), (x+w, y+h), (0, 255, 0),2)
    cv2.imshow('Car Detector', grey_scaled_frame)
    key =cv2.waitKey(1)
    if key == 81 or key == 113:
        break
# Our pretrained car classifier=
video.release()
