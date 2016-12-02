import QR
image1 = QR.stringToQRImage('kubota ryou 1029266228')
image2 = QR.transformImage(image1, 2.5, 200, 200, 4, 320, 240, True)
QR.saveImage(image2, 'sample.ppm')
