from QR import loadImage, readQRImage, stringToQRImage, saveImage
import ImageLoader

# filename = 'qr3.ppm'
# image = loadImage(filename)
# decoded = readQRImage(image)
# print(decoded)

filename2 = 'QRsample.jpg'
im = ImageLoader.loadImage(filename2)
s = readQRImage(im, strict=True)
print(s.encode('utf-8'))
# saveImage(stringToQRImage(s), 'testtt.ppm')

# tl (190.05555555555554, 377.8888888888889)
# bl (338.42, 356.0)
# tr (169.98076923076923, 225.6153846153846)
