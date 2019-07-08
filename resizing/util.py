import cv2

def resize(img,height,width):
	"""
		resizes the image to required size
		Parameters:
			height: 'New height of image'
			width: 'New width of image'
		Returns:
			resized image
	"""
	image = cv2.resize(img,(width,height),interpolation = cv2.INTER_CUBIC)
	return(image)
	
def readImg(imgPath):
	"""
		reads image from the given location
		Parameters:
			imgPath: 'Path to image'
		Returns:
			returns image
	"""
	image = cv2.imread(imgPath)
	return(image)

def RGBToBGR(img):
	"""
		converts image from RGB to BGR
		Parameters:
			img: 'image'
		Returns:
			image in BGR color code
	"""
	image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	return(image)

def BGRToRGB(img):
	"""
		converts image from BGR to RGB
		Parameters:
			img: 'image'
		Returns:
			image in RGB color code
	"""
	image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return(image)