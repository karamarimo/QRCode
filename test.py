import QR
import numpy as np

# m = np.random.rand(6, 6) > 0.5
m = np.array([[1,0,0,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0]]).astype(np.bool)
print("light%:", np.count_nonzero(m) / m.size)
print(QR._evaluateMaskedQR(m))
for row in m:
    for mo in row:
        print(' ' if mo else '0', end='')
    print('--')
