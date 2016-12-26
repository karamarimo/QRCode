import QR
import numpy as np
import time

im = QR.loadPPMImage('qr8.ppm')
print(im)

# # np.set_printoptions(threshold=np.nan)
# print(QR._stringToBits('abc'))
# # 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
# im = QR.stringToQRImage('abc')

# before = time.time()
# a = np.zeros([1000, 1000])
# for i in range(1000):
# 	for j in range(1000):
# 		a[i, j] = i * 1.1 + j * 2.3
# after = time.time()
# print(after - before, a)

# before = time.time()
# a = np.array([[i * 1.1 + j * 2.3 for j in range(1000)] for i in range(1000)])
# after = time.time()
# print(after - before, a)

# before = time.time()
# a = np.empty((1000, 1000))
# for (i, j), x in np.ndenumerate(a):
#     x = i * 1.1 + j * 2.3
# after = time.time()
# print(after - before, a)

# before = time.time()
# a = (np.arange(1000) * 1.1)
# b = np.tile(np.arange(1000) * 2.3, (1000, 1))
# a = a[:, np.newaxis] + b
# after = time.time()
# print(after - before, a)
