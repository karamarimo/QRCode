import QR
import string
import random

# random string
# s = "".join([random.choice(string.hexdigits) for i in range(53)])
# s = "Keep your friends close and your enemies closer."
s = "An apple a day keeps the doctor away."
image = QR.stringToQRImage(s, masked=False)
print(s)
QR.saveImage(image, 'test2.ppm')
