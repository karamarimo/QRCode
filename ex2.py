import QR
image = QR.stringToQRImage('An apple a day keeps the doctor away.')
QR.saveImage(image, 'qr_untransformed(An apple).ppm')
