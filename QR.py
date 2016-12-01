import math

def stringToBits(s):
	bits = "".join(["{:08b}".format(min(ord(ch), 255)) for ch in s])
	return bits

def bitsToQR(bits):
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
		raise NameError("argument bitarray is too long")

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
	return bitsToQR(stringToBits(s))

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
				# choose nearest point
				x = round(x0 + j * dxx + i * dyx)
				y = round(y0 + j * dxy + i * dyy)
				row.append(colorAt(x, y))
		newimage.append(row)
	return newimage
