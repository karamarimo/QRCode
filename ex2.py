import QR
image1 = QR.stringToQRImage('kubota ryou 1029266228')
image2 = QR.transformImage(image1, 4.0, 300, 100, 4, 320, 240, True)
QR.saveImage(image2, 'qr.ppm')