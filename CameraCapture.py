import cv2

# img = cv2.imread('qr1.png')
# print(img)
# cv2.imshow('python', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('OpenCV3 Camera Video (Press q to quit)', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
