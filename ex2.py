import QR
import string
import random

# random string
# s = "".join([random.choice(string.hexdigits) for i in range(53)])
s = "Keep your friends close and your enemies closer."
image = QR.stringToQRImage(s, masked=True)
print(s)
QR.saveImage(image, 'test1.ppm')
