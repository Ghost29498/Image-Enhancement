from blurDetection import *

if __name__=='__main__':
	modelPath = 'E:/models/naturalGrayScaleModel.h5'
	blurDetection = BlurDetection(modelPath)
	imagePath = 'E:/check.jpg'
	print(blurDetection.predict(imagePath))