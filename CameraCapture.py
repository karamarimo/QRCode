import cv2
import QR
import time
import traceback
import threading
import queue


def image_worker(in_que, out_que):
    while True:
        frame = in_que.get()
        if frame is None:
            break

        try:
            result = QR.readQRImage(frame)
            out_que.put(result)
        except ValueError:
            pass
        except Exception as e:
            pass
            raise e


winName = 'QR Capture'

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 360)

image_que = queue.Queue(maxsize=1)
result_que = queue.Queue()

t = threading.Thread(target=image_worker, args=(image_que, result_que))
t.start()

result_text = ''
info = {}
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # get a frame from the camera
    ret, frame = cap.read()

    # put frame into image_que if it's not full
    # update the frame in the queue if any
    try:
        image_que.get(block=False)
    except queue.Empty:
        pass
    try:
        image_que.put(frame, block=False)
    except queue.Full:
        pass

    # get a result string from result queue
    try:
        result_text, info = result_que.get(block=False)
    except queue.Empty:
        pass

    # draw the decoded text on the frame
    cv2.putText(frame, result_text, (20, 20), font,
                0.6, (255, 0, 0), 1, cv2.LINE_AA)

    # mark detected pdps
    if 'pdps' in info:
        for pdp in info['pdps']:
            point = tuple(int(round(x)) for x in pdp)
            point = (point[1], point[0])
            cv2.drawMarker(frame, point, (0, 0, 255),
                           markerType=cv2.MARKER_TILTED_CROSS,
                           thickness=3)

    # display the frame
    cv2.imshow(winName, frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# when quitting, release the capture, close the window,
# and stop the image processing thread
cap.release()
cv2.destroyAllWindows()

try:
    image_que.get(block=False)
except queue.Empty:
    pass
image_que.put(None)
