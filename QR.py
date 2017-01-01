import math
import numpy as np

_qr_width = 33
_quietWidth = 4
_qr_inner_width = _qr_width - _quietWidth * 2
_maxStrLen = _qr_inner_width ** 2 - 9 * 9 * 3


def _stringToBits(s):
    bits = []
    for ch in s:
        bits += [int(bit) for bit in "{:08b}".format(min(ord(ch), 255))]
    return bits


def _bitsToQR(bits):
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])

    # get color in Position Detection Pattern including margin (9*9)
    # origin is at the center
    def PDPColorAt(x, y):
        d = max(abs(x), abs(y))
        if d <= 1:
            return black
        elif d <= 2:
            return white
        elif d <= 3:
            return black
        else:
            return white

    if len(bits) > _maxStrLen:
        raise ValueError("QR error: argument bitarray is too long")

    image = np.zeros((_qr_width, _qr_width, 3), np.uint8)
    idx = 0     # index of next character of s to encode
    maxIdx = len(bits) - 1
    for i in range(_qr_width):
        for j in range(_qr_width):
            color = None
            if (i not in range(_quietWidth, _qr_width - _quietWidth)) or \
               (j not in range(_quietWidth, _qr_width - _quietWidth)):
                # quiet zone
                color = white
            elif (i in range(_quietWidth, _quietWidth + 8) and
                    j in range(_quietWidth, _quietWidth + 8)):
                # top left PDP
                color = PDPColorAt(i - (_quietWidth + 3),
                                   j - (_quietWidth + 3))
            elif (i in range(_qr_width - _quietWidth - 8,
                             _qr_width - _quietWidth) and
                    j in range(_quietWidth, _quietWidth + 8)):
                # top right PDP
                color = PDPColorAt(i - (_qr_width - _quietWidth - 4),
                                   j - (_quietWidth + 3))
            elif (i in range(_quietWidth, _quietWidth + 8) and
                    j in range(_qr_width - _quietWidth - 8,
                               _qr_width - _quietWidth)):
                # bottom left PDP
                color = PDPColorAt(i - (_quietWidth + 3),
                                   j - (_qr_width - _quietWidth - 4))
            else:
                # data
                if idx <= maxIdx:
                    color = white if bits[idx] == 1 else black
                    idx += 1
                else:
                    color = black
            image[i, j] = color
    return image


def stringToQRImage(s):
    return _bitsToQR(_stringToBits(s))


def saveImage(image, filename):
    f = open(filename, 'w')

    # header
    f.write('P3\n')                     # encoding(P3:ASCII, P6:binary)
    f.write('{0} {1}\n'.format(image.shape[1], image.shape[0]))  # width,height
    f.write('255\n')                    # max value for each RGB

    # body
    if (len(image.shape) == 3):
        # color image mode
        for row in image:
            f.write(" ".join([str(n) for n in row.flat]))
            f.write('\n')
    else:
        # binary image mode
        for row in image:
            f.write(" ".join(["0 0 0" if n == 0 else "255 255 255"
                              for n in row]))
            f.write('\n')
    f.close()


def transformImage(image, rotation, movex, movey,
                   scale, width, height, smoothing=False):
    # linearly blend 2 colors
    def blendColor(c1, c2, f):
        c = np.rint(c1 * (1 - f) + c2 * f).astype(np.uint8)
        return c

    backgroundColor = np.array([128, 128, 128])
    orgWidth = image.shape[1]
    orgHeight = image.shape[0]

    # get the color at v(x,y) of the original image
    def colorAt(x, y):
        if x in range(orgWidth) and y in range(orgHeight):
            return image[y, x]
        else:
            return backgroundColor

    # axis vector and origin point
    conv = np.array([[math.cos(-rotation), -math.sin(-rotation)],
                     [math.sin(-rotation), math.cos(-rotation)]]) / scale
    v0 = conv @ [-movey, -movex]

    newimage = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            v = v0 + conv @ np.array([i, j])
            if smoothing:
                # linear interpolation
                if (v[1] >= -1 and v[1] <= orgWidth and
                        v[0] >= -1 and v[0] <= orgHeight):
                    x1 = math.floor(v[1])
                    x2 = x1 + 1
                    y1 = math.floor(v[0])
                    y2 = y1 + 1
                    c1 = blendColor(colorAt(x1, y1), colorAt(x2, y1),
                                    v[1] - x1)
                    c2 = blendColor(colorAt(x1, y2), colorAt(x2, y2),
                                    v[1] - x1)
                    c = blendColor(c1, c2, v[0] - y1)
                    newimage[i, j] = c
                else:
                    newimage[i, j] = backgroundColor
            else:
                # nearest neighbor interpolation
                v1 = np.rint(v).astype(int)
                newimage[i, j] = colorAt(v1[1], v1[0])
    return newimage


# reads a ppm file and returns a image object(2D ndarray)
def loadPPMImage(filename):
    def formatError():
        f.close()
        raise ValueError('QR error: file has wrong format')

    try:
        f = open(filename, 'r')
    except Exception as e:
        raise ValueError('QR error: cannot open file')

    # get header
    stack = []
    while True:
        line = f.readline()
        sharpIdx = line.find('#')
        if sharpIdx > -1:
            # remove the string after '#'
            line = line[0:sharpIdx]
        if line == '':
            # if it reached EOF
            formatError()

        stack += line.split()
        if len(stack) >= 4:
            # when all the header is read
            break

    # check header
    magicNum = stack.pop(0)
    if magicNum != 'P3':
        formatError()

    try:
        width = int(stack.pop(0))
        height = int(stack.pop(0))
        maxVal = int(stack.pop(0))
    except Exception as e:
        formatError()

    if maxVal != 255:
        formatError()

    # get data
    while True:
        line = f.readline()
        stack += line.split()
        if line == '':
            # if it reached EOF
            break
    f.close()

    # if the number of color values is not right
    if len(stack) != width * height * 3:
        formatError()

    image = np.array(stack, np.uint8)
    image = image.reshape((height, width, 3))
    return image


# binarize image (returned image's color is 0 or 1)
def _binarizeImage(image):
    # threshold = average of R + G + B
    threshold = image.mean(axis=(0, 1)).sum()
    colorSum = image.sum(axis=2)    # sum of RGB at each pixel
    _image = np.where(colorSum < threshold, 0, 1)
    return _image


# take a ndarray of 0,1 and encode it into a list using run-length encoding
# ex. array([0,0,1,1,1,0]) => [(0,2),(1,3),(0,1)]
def _runLengthEncode(list):
    if list.size == 0:
        return []

    out = []
    last = list[0]
    runLength = 0

    for b in list:
        if b == last:
            runLength += 1
        else:
            out.append((last, runLength))
            last = b
            runLength = 1
    out.append((last, runLength))
    return out


# take a ndarrayof 0,1 and
# return the center of potential position detection patterns
def _findPDP(array, strict):
    def near(value, origin, error):
        return abs(value - origin) < error

    rlelist = _runLengthEncode(array)
    out = []

    # find 1B:1W:3B:1B:1W patterns
    for i in range(2, len(rlelist) - 2):
        if rlelist[i][0] == 0 and rlelist[i][1] >= 3:
            mod_width = rlelist[i][1] / 3
            # allowable error [pixel]
            tolerance = 4 / 3 if strict else mod_width / 8
            if (near(rlelist[i - 1][1], mod_width, tolerance) and
                    near(rlelist[i - 2][1], mod_width, tolerance) and
                    near(rlelist[i + 1][1], mod_width, tolerance) and
                    near(rlelist[i + 2][1], mod_width, tolerance)):
                # get the index of the center of
                # the detected pattern in the original list
                start = sum(ele[1] for ele in rlelist[:i - 2])
                length = sum(ele[1] for ele in rlelist[i - 2:i + 3])
                center = start + (length - 1) / 2
                out.append(center)
    return out


def _angle(v1, v2):
    if (not v1.any()) or (not v2.any()):
        raise ValueError("invalid argument")
    cos = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    return math.acos(cos)


def _crossProduct(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


# decode a list of {0,1} into a string (unicode, utf-8)
def _decode(code):
    out = ""
    for i in range(0, len(code), 8):
        codePoint = 0
        for j in range(i, i + 8):
            codePoint = codePoint * 2 + code[j]
        cha = chr(codePoint)
        out += cha
    return out


def _readBinaryQRImage(binaryImage, strict):
    def removeArray(lis, item):
        for i, ele in enumerate(lis):
            if np.array_equal(item, ele):
                lis.pop(i)
                break

    # center points of potential position detection patterns
    points = []
    # check rows
    for y in range(binaryImage.shape[0]):
        patterns = _findPDP(binaryImage[y], strict)
        points += [np.array([y, x, 0]) for x in patterns]
    # check columns
    for x in range(binaryImage.shape[1]):
        patterns = _findPDP(binaryImage[:, x], strict)
        points += [np.array([y, x, 1]) for y in patterns]

    # split into clusters
    clusters = []
    while len(points) > 0:
        cluster = [points.pop()]
        while True:
            neighbors = [
                p for p in points
                if min(np.linalg.norm((p - p2)[:2])
                       for p2 in cluster) < 1.5
            ]
            if len(neighbors) == 0:
                break
            else:
                for neighbor in neighbors:
                    removeArray(points, neighbor)
                cluster += neighbors
        clusters.append(cluster)

    # only accept clusters that have points from both RLE direction
    validClusters = []
    for cluster in clusters:
        k = sum(p[2] for p in cluster)
        if k > 0 and k < len(cluster):
            validClusters.append(cluster)

    # not enough detected patterns
    if len(validClusters) < 3:
        raise ValueError("QR error: failed to read QR image")

    # select top 3 biggest clusters
    validClusters.sort(key=len)
    validClusters = validClusters[-3:]

    # get average point of each cluster
    pdps = []
    for cluster in validClusters:
        avg = sum(cluster) / len(cluster)
        pdps.append(avg)

    # get which angle is the nearest to 90 degree
    angles = []
    for i in range(3):
        j = (i + 1) % 3
        k = (j + 1) % 3
        v1 = pdps[j] - pdps[i]
        v2 = pdps[k] - pdps[i]
        angle1 = _angle(v1, v2)
        angles.append(abs(angle1 - math.pi / 2))
    tl = angles.index(min(angles))
    # determine which point is which PDP
    pdp_tl = pdps[tl]

    i1 = (tl + 1) % 3
    i2 = (tl + 2) % 3
    v1 = pdps[i1] - pdps[tl]
    v2 = pdps[i2] - pdps[tl]
    if _crossProduct(v1, v2) > 0:
        pdp_bl = pdps[i1]
        pdp_tr = pdps[i2]
    else:
        pdp_bl = pdps[i2]
        pdp_tr = pdps[i1]

    # print("topleft pdp:", pdp_tl)
    # print("bottomleft pdp:", pdp_bl)
    # print("topright pdp", pdp_tr)

    # get the origin coordinates of QR pattern (excluding quiet zone)
    d = _qr_inner_width - 3 * 2 - 1
    dy = (pdp_bl - pdp_tl) / d
    dx = (pdp_tr - pdp_tl) / d
    conv = np.array([dy, dx]).T
    v0 = pdp_tl + conv @ np.array([-3, -3])
    # print('v0:', v0)

    # check if the bottom right corner is in image
    bottomright = v0 + (dx + dy) * _qr_inner_width
    if not (bottomright[0] > 0 and bottomright[0] < binaryImage.shape[0] and
            bottomright[1] > 0 and bottomright[1] < binaryImage.shape[1]):
        raise ValueError("QR error: failed to read QR image")

    # get code from image
    code = []
    for j in range(_qr_inner_width):
        for i in range(_qr_inner_width):
            # ignore position detection patterns
            if ((i in range(8) and j in range(8)) or
                (i in range(_qr_inner_width - 8, _qr_inner_width) and
                    j in range(8)) or
                (i in range(8) and
                    j in range(_qr_inner_width - 8, _qr_inner_width))):
                continue
            v = v0 + conv @ np.array([j, i])
            # get module at (x,y) using bilinear interpolation
            x_f = math.floor(v[1])
            x_c = x_f + 1
            y_f = math.floor(v[0])
            y_c = y_f + 1
            m1 = (x_c - v[1]) * binaryImage[y_f, x_f] + \
                (v[1] - x_f) * binaryImage[y_f, x_c]
            m2 = (x_c - v[1]) * binaryImage[y_c, x_f] + \
                (v[1] - x_f) * binaryImage[y_c, x_c]
            m = (y_c - v[0]) * m1 + (v[0] - y_f) * m2
            m = round(m).astype(int)
            code.append(m)
    # print(code)

    # trim trailing 00..0 off the code into a length of a multple of 8
    code = code[:-(len(code) % 8)]
    while len(code) > 8:
        if code[-8:] == [0] * 8:
            code = code[:-8]
        else:
            break
    decoded = _decode(code)
    return decoded


def readQRImage(image, strict=True):
    return _readBinaryQRImage(_binarizeImage(image), strict)
