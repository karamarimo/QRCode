from PIL import Image


def loadImage(filename):
    im = Image.open(filename)
    array = list(im.getdata())
    width = im.width
    height = im.height
    image = []
    for i in range(height):
        image.append(array[width * i:width * (i + 1)])
    return image
