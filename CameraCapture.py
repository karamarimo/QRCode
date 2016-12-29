import cv2
import QR
import time

# img = cv2.imread('qr1.png')
# print(img)
# cv2.imshow('python', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
counter = 0
readInterval = 300
while(True):
    counter -= 1
    ret, frame = cap.read()
    string = ''
    if counter <= 0:
        counter = readInterval
        try:
            before = time.time()
            result = QR.readQRImage(frame)
            print(result.encode('utf-8'))
            after = time.time()
            print('time: ', after - before)
        except Exception as e:
            print(e.args)
        # print('result: ', result)

    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(frame, string, (10, 500), font,
                1, (150, 255, 150), 5, cv2.LINE_AA)
    # Display the resulting frame
    # cv2.imshow('OpenCV3 Camera Video (Press q to quit)', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
