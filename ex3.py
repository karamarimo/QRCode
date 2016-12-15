import QR
image1 = QR.stringToQRImage("A chain is only as strong as its weakest link.")
image2 = QR.transformImage(image1, 4.1, 230, 150, 2, 320, 240, False)
QR.saveImage(image2, 'qr8.ppm')
