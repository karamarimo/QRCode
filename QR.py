def stringToBits(s, encoding='utf-8'):
	if encoding == 'utf-8':
		bits = "".join(["{:08b}".format(min(ord(ch), 255)) for ch in s])
		return bits
	elif encoding == 'utf-16':
		bits = "".join(["{:016b}".format(ord(ch)) for ch in s])
		return bits		
	else:
		raise NameError("invalid argument for encoding")

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
		print('string length is too long')
		return []

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

def saveQRImage(qr, filename):
	f = open(filename, 'w')

	# header
	f.write('P3\n')						# encoding(P3:ASCII, P6:binary)
	f.write('{0} {1}\n'.format(len(qr), len(qr[0])))	# width, height
	f.write('255\n')					# max value for each RGB

	# body
	for row in qr:
		for module in row:
			f.write(" ".join([str(n) for n in module]) + " ")
		f.write('\n')
	f.close()

def createQRImageFromString(s, filename):
	saveQRImage(bitsToQR(stringToBits(s)), filename)
