import math
import numpy as np
from itertools import chain

qr_width = 33
quietWidth = 4
qr_inner_width = qr_width - quietWidth * 2
maxBitLen = qr_inner_width ** 2 - 8 * 8 * 3


# take a ndarray and encode it into a list using run-length encoding
# ex. array([0,0,1,1,1,0]) => [(0,2),(1,3),(0,1)]
def _runLengthEncode(array):
    if array.size == 0:
        return []

    out = []
    last = array[0]
    runLength = 0

    for b in array:
        if b == last:
            runLength += 1
        else:
            out.append((last, runLength))
            last = b
            runLength = 1
    out.append((last, runLength))
    return out


def _stringToBits(s):
    bits = []
    for ch in s:
        bits += [int(bit) for bit in "{:08b}".format(min(ord(ch), 255))]
    return bits


def _getMaskFun(id):
    if id == 0:
        return lambda y, x: (y + x) % 2 == 0
    elif id == 1:
        return lambda y, x: y % 2 == 0
    elif id == 2:
        return lambda y, x: x % 3 == 0
    elif id == 3:
        return lambda y, x: (y + x) % 3 == 0
    elif id == 4:
        return lambda y, x: (y // 2 + x // 3) % 2 == 0
    elif id == 5:
        return lambda y, x: (y * x) % 2 + (y * x) % 3 == 0
    elif id == 6:
        return lambda y, x: ((y * x) % 2 + (y * x) % 3) % 2 == 0
    elif id == 7:
        return lambda y, x: ((y + x) % 2 + (y * x) % 3) % 2 == 0
    else:
        ValueError("invalid mask id")


def _getMask(id):
    mf = _getMaskFun(id)
    m = np.zeros((qr_inner_width, qr_inner_width), np.bool)
    for y in range(qr_inner_width):
        for x in range(qr_inner_width):
            m[y, x] = mf(y, x)

    # clear the parts for function patterns
    m[:8, :8] = False
    m[-8:, :8] = False
    m[:8, -8:] = False

    return m


def _evaluateMaskedQR(qrimage):
    lines = chain((row for row in qrimage), (col for col in qrimage.T))

    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    for line in lines:
        rle = _runLengthEncode(line)

        # find consective modules of length more than 5
        # (length - 2) points per each
        for run in rle:
            if run[1] >= 5:
                score1 += run[1] - 2

        # find patterns of 0100010 next to 1111
        # 40 points per each
        length = len(rle)
        if length >= 6:
            for i in range(2, length - 2):
                if (rle[i] == (False, 3)):
                    if ((i + 3 < length and
                         rle[i - 1][1] == 1 and
                         rle[i + 1][1] == 1 and
                         rle[i + 2][1] == 1 and
                         rle[i + 3][1] >= 4) or
                        (i - 3 >= 0 and
                         rle[i + 1][1] == 1 and
                         rle[i - 1][1] == 1 and
                         rle[i - 2][1] == 1 and
                         rle[i - 3][1] >= 4)):
                        score2 += 40

    # find 2*2 squares of the same modules
    # 3 points per each
    h = qrimage.shape[0]
    w = qrimage.shape[1]

    for y in range(h - 1):
        for x in range(w - 1):
            m = qrimage[y, x]
            if (m == qrimage[y + 1, x] and
                    m == qrimage[y, x + 1] and
                    m == qrimage[y + 1, x + 1]):
                score3 += 3

    # when light modules' percentage is p and
    # 5k <= |50 - p| < 5(k + 1),
    # 10 points per k
    p = np.count_nonzero(qrimage) / qrimage.size * 100
    k = int(math.floor(abs(p - 50) / 5))
    score4 += k * 10

    print(score1, score2, score3, score4)
    return score1 + score2 + score3 + score4


def _bitsToQRImage(bits, masked):
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])

    _bits = ([0, 0, 0] + bits) if masked else bits

    # get color in Position Detection Pattern including margin (9*9)
    # origin is at the center
    def PDPModuleAt(x, y):
        d = max(abs(x), abs(y))
        if d <= 1:
            return False
        elif d <= 2:
            return True
        elif d <= 3:
            return False
        else:
            return True

    if len(_bits) > maxBitLen:
        raise ValueError("QR error: argument bitarray is too long")

    # create qr code without quiet zone
    image = np.zeros((qr_inner_width, qr_inner_width), dtype=np.bool)
    idx = 0     # index of next character of s to encode
    maxIdx = len(_bits) - 1
    crange = range(qr_inner_width)
    for i in crange:
        for j in crange:
            if (i in crange[:8] and j in crange[:8]):
                # top left PDP
                module = PDPModuleAt(i - 3, j - 3)
            elif (i in crange[-8:] and j in crange[:8]):
                # bottom left PDP
                module = PDPModuleAt(i - (qr_inner_width - 1 - 3), j - 3)
            elif (i in crange[:8] and j in crange[-8:]):
                # top right PDP
                module = PDPModuleAt(i - 3, j - (qr_inner_width - 1 - 3))
            else:
                # data
                if idx <= maxIdx:
                    module = True if _bits[idx] == 1 else False
                    idx += 1
                else:
                    module = False
            image[i, j] = module

    if masked:
        maskedCodes = [np.logical_xor(_getMask(i), image) for i in range(8)]
        scores = [_evaluateMaskedQR(code) for code in maskedCodes]
        best = scores.index(min(scores))
        maskbits = [int(x) for x in "{:03b}".format(best)]
        image = maskedCodes[best]
        print('selected mask pattern:', maskbits)
        # put mask pattern bits to the first 3 data modules
        image[0, 8:11] = maskbits

    # add quiet zone
    image = np.pad(image, pad_width=quietWidth,
                   mode='constant', constant_values=True)

    # convert each module into a color(black/white)
    image = np.where(image[:, :, np.newaxis], white, black)
    return image


def stringToQRImage(s, masked=False):
    return _bitsToQRImage(_stringToBits(s), masked)


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
            # if mod_width < 5:
            #     continue
            # allowable error [pixel]
            tolerance = 6 / 3 if strict else mod_width / 8
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


def _readBinaryQRImage(binaryImage, masked, strict):
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

    # error if not enough pdps detected
    if len(validClusters) < 3:
        raise ValueError("QR error: cannot detect position detection patterns")

    # select top 3 biggest clusters
    validClusters.sort(key=len)
    validClusters = validClusters[-3:]
    # print([len(cl) for cl in validClusters])
    # get average point of each cluster
    pdps = []
    for cluster in validClusters:
        avg = sum(cluster) / len(cluster)
        avg = avg[:2]       # remove the third field
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
    d = qr_inner_width - 3 * 2 - 1
    dy = (pdp_bl - pdp_tl) / d
    dx = (pdp_tr - pdp_tl) / d
    conv = np.array([dy, dx]).T
    v0 = pdp_tl + conv @ np.array([-3, -3])
    # print('v0:', v0)

    # check if the four corners are all in the image
    for j in range(2):
        for i in range(2):
            corner = v0 + conv @ np.array(
                [j * qr_inner_width, i * qr_inner_width])
            if not (corner[0] > 0 and corner[0] < binaryImage.shape[0] and
                    corner[1] > 0 and corner[1] < binaryImage.shape[1]):
                raise ValueError("QR error: qr code is partly off the image")

    # get code from image
    code = []
    maskFun = None
    codeCount = 0
    for j in range(qr_inner_width):
        for i in range(qr_inner_width):
            # ignore position detection patterns
            if ((i in range(8) and j in range(8)) or
                (i in range(qr_inner_width - 8, qr_inner_width) and
                    j in range(8)) or
                (i in range(8) and
                    j in range(qr_inner_width - 8, qr_inner_width))):
                continue

            codeCount += 1
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

            if masked:
                if codeCount > 3:
                    # unmask
                    mask = maskFun(j, i)
                    m = 1 - m if mask == 1 else m
                    code.append(m)
                else:
                    code.append(m)
                    # get mask function with the first 3 bit code
                    if codeCount == 3:
                        maskId = code[0] * 4 + code[1] * 2 + code[2]
                        print(maskId)
                        maskFun = _getMaskFun(maskId)
                        code.clear()
            else:
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

    info = {'pdps': pdps}
    return (decoded, info)


def readQRImage(image, masked=False, strict=True):
    return _readBinaryQRImage(_binarizeImage(image), masked, strict)
