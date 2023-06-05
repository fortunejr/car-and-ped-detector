import cv2

classifier_file = 'cardetector.xml'
# Our Image
img_file = 'C:\\Users\\user\\Pictures\\carimg.PNG'
video = cv2.VideoCapture('dashcam3.MOV')

while True:
    read_successful, frame = video.read()
    
    if read_successful:
        grey_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        break
    car_traker = cv2.CascadeClassifier(classifier_file)
    cars = car_traker.detectMultiScale(grey_scaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(grey_scaled_frame, (x, y), (x+w, y+h), (0,0,255),2)
    cv2.imshow('Car Detector', grey_scaled_frame)
    cv2.waitKey(1)
# Our pretrained car classifier
'''classifier_file = 'cardetector.xml'
img = cv2.imread(img_file)
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
car_traker = cv2.CascadeClassifier(classifier_file)
cars = car_traker.detectMultiScale(black_and_white)

for (x,y,w,h) in cars:
    cv2.rectangle(black_and_white, (x, y), (x+w, y+h), (0,0,255),2)
print(cars)
cv2.imshow('Car Detector', black_and_white)
cv2.waitKey()
print('Code Completed')'''