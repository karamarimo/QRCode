from QR import loadPPMImage, readQRImage, stringToQRImage, saveImage, _stringToBits, _binarizeImage
import ImageLoader

filename = 'qr5.ppm'
image = loadPPMImage(filename)
decoded = readQRImage(image)
print(decoded.encode('utf-8'))

# filename2 = 'QRsample.jpg'
# im = ImageLoader.loadImage(filename2)
# s = readQRImage(im, strict=True)
# print(s.encode('utf-8'))
# # saveImage(stringToQRImage(s), 'testtt.ppm')

# filename = 'qr10.ppm'
# im = loadPPMImage(filename)
# bim = _binarizeImage(im)
# saveImage(bim, 'qr10_b.ppm')
