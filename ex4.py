from QR import loadImage, readQRImage

filename = 'qr7.ppm'
image = loadImage(filename)
decoded = readQRImage(image)
print(decoded)
# QR.saveImage(bimage, "binary_" + filename)
