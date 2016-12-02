import math

def __stringToBits(s):
	bits = "".join(["{:08b}".format(min(ord(ch), 255)) for ch in s])
	return bits

def __bitsToQR(bits):
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

	width = 33
	quietWidth = 4
	maxStrLen = (width - quietWidth * 2) ** 2 - 9 * 9 * 3

	if len(bits) > maxStrLen:
		raise NameError("QR error: argument bitarray is too long")

	rows = []
	idx = 0	# index of next character of s to encode
	maxIdx = len(bits) - 1
	for i in range(width):
		row = []
		for j in range(width):
			color = None
			if (i not in range(quietWidth, width - quietWidth)) or \
				(j not in range(quietWidth, width - quietWidth)):
				# quiet zone
				color = white
			elif i in range(quietWidth, quietWidth + 8) and \
				j in range(quietWidth, quietWidth + 8):
				# top left PDP
				color = PDPColorAt(i - (quietWidth + 3), j - (quietWidth + 3))
			elif i in range(width - quietWidth - 8, width - quietWidth) and \
				j in range(quietWidth, quietWidth + 8):
				# top right PDP
				color = PDPColorAt(i - (width - quietWidth - 4), j - (quietWidth + 3))
			elif i in range(quietWidth, quietWidth + 8) and \
				j in range(width - quietWidth - 8, width - quietWidth):
				# bottom left PDP
				color = PDPColorAt(i - (quietWidth + 3), j - (width - quietWidth - 4))
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
	return __bitsToQR(__stringToBits(s))

def saveImage(image, filename):
	f = open(filename, 'w')

	# header
	f.write('P3\n')						# encoding(P3:ASCII, P6:binary)
	f.write('{0} {1}\n'.format(len(image[0]), len(image)))	# width, height
	f.write('255\n')					# max value for each RGB

	# body
	for row in image:
		for module in row:
			f.write(" ".join([str(n) for n in module]) + " ")
		f.write('\n')
	f.close()

def transformImage(image, rotation, movex, movey, scale, width, height, smoothing=True):
	def blendColor(c1, c2, f):
		c = [round(c1[i] * (1 - f) + c2[i] * f) for i in range(3)]
		return (c[0], c[1], c[2])

	backgroundColor = (0, 0, 255)
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
					c1 = blendColor(colorAt(x1,y1), colorAt(x2,y1), x - x1)
					c2 = blendColor(colorAt(x1,y2), colorAt(x2,y2), x - x1)
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
def loadImage(filename):
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
			line = line[0 : sharpIdx]
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
def binarizeImage(image):
	width = len(image[0])
	height = len(image)

	# threshold = average of R + G + B
	threshold = sum(sum(sum(color) for color in row) for row in image) / (width * height)
	print('threshold = ' + str(threshold))
	out = []
	for row in image:
		orow = []
		for color in row:
			c = 0 if sum(color) < threshold else 1
			orow.append(c)
		out.append(orow)
	return out
