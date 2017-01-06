import QR
import numpy as np
import time
import threading
import queue


def doSome(in_que, out_que):
    while True:
        x = in_que.get()
        time.sleep(1)
        result = x + 100
        out_que.put(result)


image_que = queue.Queue(maxsize=1)
result_que = queue.Queue()


t = threading.Thread(target=doSome, args=(image_que, result_que))
t.start()

i = 0
while True:
    try:
        image_que.put(i, block=False)
    except queue.Full as e:
        pass
    try:
        re = result_que.get(block=False)
        print(re)
    except queue.Empty as e:
        pass
    time.sleep(0.5)
    i += 1

