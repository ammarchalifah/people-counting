import cv2

vs = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = vs.read()

    if ret is None:
        break
    
    cv2.imshow('video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break

vs.release()
cv2.destroyAllWindows()