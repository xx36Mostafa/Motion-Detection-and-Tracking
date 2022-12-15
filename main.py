import imutils
import cv2
import numpy as np

FRAMES_TO_PERSIST = 10

MIN_SIZE_FOR_MOVEMENT = 2000


MOVEMENT_DETECTED_PERSISTENCE = 100


cap = cv2.VideoCapture(5) # Flush the stream
cap.release()
cap = cv2.VideoCapture(0) # Then start the webcam

first_frame = None
next_frame = None

font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0

while True:

    transient_movement_flag = False
    
    ret, frame = cap.read()
    text = "Unoccupied"

    if not ret:
        print("CAPTURE ERROR")
        continue

    frame = imutils.resize(frame, width = 750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None: first_frame = gray    

    delay_counter += 1

    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

        
    next_frame = gray

    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations = 2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)
        

        if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            transient_movement_flag = True
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # movement timer.
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
  
    if movement_persistent_counter > 0:
        text = "Movement Detected " + str(movement_persistent_counter)
        movement_persistent_counter -= 1
    else:
        text = "No Movement Detected"


    cv2.putText(frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
    

    
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

    cv2.imshow("frame", np.hstack((frame_delta, frame)))


    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

# Cleanup when closed
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
