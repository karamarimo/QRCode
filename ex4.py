import QR
import time

image = QR.loadImage('qr2.ppm')

bimage = QR.binarizeImage(image)
