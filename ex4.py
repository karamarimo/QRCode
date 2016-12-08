import QR
import time

filename = 'qr1.ppm'
image = QR.loadImage(filename)
bimage = QR.binarizeImage(image)
decoded = QR.readQR(bimage)
print(decoded)
# QR.saveImage(bimage, "binary_" + filename)
