import QR
import string
import random

# random string
s = "".join([random.choice(string.hexdigits) for i in range(53)])
image = QR.stringToQRImage(s, masked=True)
print(s)
QR.saveImage(image, 'random_string.ppm')
