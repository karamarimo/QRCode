from QR import loadPPMImage, readQRImage, stringToQRImage, saveImage
import ImageLoader

filename = 'qr3.ppm'
image = loadPPMImage(filename)
decoded = readQRImage(image)
print(decoded)

filename2 = 'QRsample.jpg'
im = ImageLoader.loadImage(filename2)
s = readQRImage(im, strict=True)
print(s.encode('utf-8'))
# saveImage(stringToQRImage(s), 'testtt.ppm')

