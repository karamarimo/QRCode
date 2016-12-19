import math

_qr_width = 33
_quietWidth = 4
_qr_inner_width = _qr_width - _quietWidth * 2
_maxStrLen = _qr_inner_width ** 2 - 9 * 9 * 3


def _stringToBits(s):
    bits = "".join(["{:08b}".format(min(ord(ch), 255)) for ch in s])
    return bits


def _bitsToQR(bits):
    black = (0, 0, 0)
    white = (255, 255, 255)

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
        raise NameError("QR error: argument bitarray is too long")

    rows = []
    idx = 0     # index of next character of s to encode
    maxIdx = len(bits) - 1
    for i in range(_qr_width):
        row = []
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
                    color = white if bits[idx] == '1' else black
                    idx += 1
                else:
                    color = black
            row.append(color)
        rows.append(row)
    return rows


def stringToQRImage(s):
    return _bitsToQR(_stringToBits(s))


def saveImage(image, filename):
    f = open(filename, 'w')

    # header
    f.write('P3\n')                     # encoding(P3:ASCII, P6:binary)
    f.write('{0} {1}\n'.format(len(image[0]), len(image)))  # width, height
    f.write('255\n')                    # max value for each RGB

    # body
    if (type(image[0][0]) == tuple):
        # color image mode
        for row in image:
            for module in row:
                f.write(" ".join([str(n) for n in module]) + " ")
            f.write('\n')
    else:
        # binary image mode
        for row in image:
            for module in row:
                color = "0 0 0" if module == 0 else "255 255 255"
                f.write(color + " ")
            f.write('\n')
    f.close()


def transformImage(image, rotation, movex, movey,
                   scale, width, height, smoothing=False):
    # linearly blend 2 colors
    def blendColor(c1, c2, f):
        c = [round(c1[i] * (1 - f) + c2[i] * f) for i in range(3)]
        return (c[0], c[1], c[2])

    backgroundColor = (128, 128, 128)
    orgWidth = len(image[0])
    orgHeight = len(image)

    # get the color at (x,y) of the original image
    def colorAt(x, y):
        if x in range(orgWidth) and y in range(orgHeight):
            return image[y][x]
        else:
            return backgroundColor

    # reading vector and origin point
    dxx = math.cos(-rotation) / scale
    dxy = -math.sin(-rotation) / scale
    dyx = -dxy
    dyy = dxx
    x0 = ((-movex) * dxx + (-movey) * dyx)
    y0 = ((-movex) * dxy + (-movey) * dyy)

    newimage = []
    for i in range(height):
        row = []
        for j in range(width):
            if smoothing:
                # linear interpolation
                x = x0 + j * dxx + i * dyx
                y = y0 + j * dxy + i * dyy
                if x >= -1 and x <= orgWidth and y >= -1 and y <= orgHeight:
                    x1 = math.floor(x)
                    x2 = x1 + 1
                    y1 = math.floor(y)
                    y2 = y1 + 1
                    c1 = blendColor(colorAt(x1, y1), colorAt(x2, y1), x - x1)
                    c2 = blendColor(colorAt(x1, y2), colorAt(x2, y2), x - x1)
                    c = blendColor(c1, c2, y - y1)
                    row.append(c)
                else:
                    row.append(colorAt(x, y))
            else:
                # nearest neighbor interpolation
                x = round(x0 + j * dxx + i * dyx)
                y = round(y0 + j * dxy + i * dyy)
                row.append(colorAt(x, y))
        newimage.append(row)
    return newimage


# reads a ppm file and returns a image object(2D array)
def loadPPMImage(filename):
    def formatError():
        f.close()
        raise NameError('QR error: file has wrong format')

    try:
        f = open(filename, 'r')
    except Exception as e:
        raise NameError('QR error: cannot open file')

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

    # get data
    image = []
    while True:
        line = f.readline()
        stack += line.split()
        if line == '':
            # if it reached EOF
            break

    # if the number of color values is not right
    if len(stack) != width * height * 3:
        formatError()

    try:
        for i in range(height):
            row = []
            for j in range(width):
                start = (i * width + j) * 3
                end = start + 3
                color = tuple(int(s) for s in stack[start:end])
                for val in color:
                    if val > maxVal:
                        formatError()
                row.append(color)
            image.append(row)
    except Exception as e:
        formatError()

    f.close()
    return image


# binarize image (returned image's color is 0 or 1)
def _binarizeImage(image):
    width = len(image[0])
    height = len(image)

    # threshold = average of R + G + B
    threshold = sum(sum(sum(color) for color in row) for row in image) \
        / (width * height)
    print('threshold = ' + str(threshold))
    out = []
    for row in image:
        orow = []
        for color in row:
            c = 0 if sum(color) < threshold else 1
            orow.append(c)
        out.append(orow)
    return out


# take a list of 0,1 and encode it using run-length encoding
# ex. [0,0,1,1,1,0] => [(0,2),(1,3),(0,1)]
def _runLengthEncode(list):
    if len(list) == 0:
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


# take a run-length encoded list and find position detection patterns
def _findPDP(list, strict):
    def near(value, origin, error):
        return abs(value - origin) < error

    rlelist = _runLengthEncode(list)
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


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _angle(v1, v2):
    if v1 == (0, 0) or v2 == (0, 0):
        raise NameError("invalid argument")
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) \
        / (math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1]))
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
    # center points of potential position detection patterns
    points = []
    # check rows
    for y in range(len(binaryImage)):
        patterns = _findPDP(binaryImage[y], strict)
        points += [(x, y) for x in patterns]
    # check columns
    for x in range(len(binaryImage[0])):
        patterns = _findPDP([row[x] for row in binaryImage], strict)
        points += [(x, y) for y in patterns]

    # split into clusters
    clusters = []
    while len(points) > 0:
        cluster = [points.pop()]
        while True:
            neighbors = [p for p in points
                         if min(_distance(p, p2) for p2 in cluster) < 1.5]
            if len(neighbors) == 0:
                break
            else:
                for neighbor in neighbors:
                    points.remove(neighbor)
                cluster += neighbors
        clusters.append(cluster)
    # for cluster in clusters:
    #   print("cluster: ", cluster)

    # not enough detected patterns
    if len(clusters) < 3:
        raise NameError("QR error: failed to read QR image")

    # select top 3 biggest clusters
    clusters.sort(key=len)
    clusters = clusters[-3:]

    # get average point of each cluster
    pdps = []
    for cluster in clusters:
        x = sum(p[0] for p in cluster) / len(cluster)
        y = sum(p[1] for p in cluster) / len(cluster)
        pdps.append((x, y))

    # get which angle is the nearest to 90 degree
    angles = []
    for i in range(3):
        j = (i + 1) % 3
        k = (j + 1) % 3
        v1 = (pdps[j][0] - pdps[i][0], pdps[j][1] - pdps[i][1])
        v2 = (pdps[k][0] - pdps[i][0], pdps[k][1] - pdps[i][1])
        angle1 = _angle(v1, v2)
        angles.append(abs(angle1 - math.pi / 2))
    tl = angles.index(min(angles))
    # determine which point is which PDP
    pdp_tl = pdps[tl]

    i1 = (tl + 1) % 3
    i2 = (tl + 2) % 3
    v1 = (pdps[i1][0] - pdps[tl][0], pdps[i1][1] - pdps[tl][1])
    v2 = (pdps[i2][0] - pdps[tl][0], pdps[i2][1] - pdps[tl][1])
    if _crossProduct(v1, v2) > 0:
        pdp_bl = pdps[i2]
        pdp_tr = pdps[i1]
    else:
        pdp_bl = pdps[i1]
        pdp_tr = pdps[i2]

    print("tl", pdp_tl)
    print("bl", pdp_bl)
    print("tr", pdp_tr)

    # get the origin coordinates of QR pattern (excluding quiet zone)
    d = _qr_inner_width - 3 * 2 - 1
    dx_x = (pdp_tr[0] - pdp_tl[0]) / d
    dx_y = (pdp_tr[1] - pdp_tl[1]) / d
    dy_x = (pdp_bl[0] - pdp_tl[0]) / d
    dy_y = (pdp_bl[1] - pdp_tl[1]) / d
    x0 = pdp_tl[0] - 3 * dx_x - 3 * dy_x
    y0 = pdp_tl[1] - 3 * dx_y - 3 * dy_y

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

            x = x0 + dx_x * i + dy_x * j
            y = y0 + dx_y * i + dy_y * j
            # get module at (x,y) using bilinear interpolation
            x_f = math.floor(x)
            x_c = x_f + 1
            y_f = math.floor(y)
            y_c = y_f + 1
            m1 = (x_c - x) * binaryImage[y_f][x_f] + \
                (x - x_f) * binaryImage[y_f][x_c]
            m2 = (x_c - x) * binaryImage[y_c][x_f] + \
                (x - x_f) * binaryImage[y_c][x_c]
            m = (y_c - y) * m1 + (y - y_f) * m2
            m = round(m)
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
