import cv2
import QR
import time
import traceback

# img = cv2.imread('qr1.png')
# print(img)
# cv2.imshow('python', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

counter = 0
readInterval = 1
result = ''
while(True):
    counter -= 1
    ret, frame = cap.read()
    if counter <= 0:
        try:
            counter = readInterval
            before = time.time()
            result = QR.readQRImage(frame)
            after = time.time()
            # print(result.encode('utf-8'))
            # print('time: ', after - before)
            # break
        except Exception as e:
            # traceback.print_exc()
            pass
        # print('result: ', result)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result, (20, 20), font,
                0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('OpenCV3 Camera Video (Press q to quit)', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
