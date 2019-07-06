import numpy as np
import cv2
"""
  Produces blur images
"""
def softBlurContours(image, contours, ksize, sigmaX, *args, **kwargs):
	iterations = 3
	if 'iters' in kwargs:
		iterations = kwargs['iters']
	sigmaY = args[0] if len(args) > 0 and args[0] != None else sigmaX
	mksize = args[1] if len(args) > 1 and args[1] != None else ksize
	msigmaX = args[2] if len(args) > 2 and args[2] != None else sigmaX
	msigmaY = args[3] if len(args) > 3 and args[3] != None else msigmaX
	mask = np.zeros(image.shape[:2])
	for i, contour in enumerate(contours):
		cv2.drawContours(mask, contour, 0, 255, -1)
	blurred_image = cv2.GaussianBlur(image, (ksize,ksize), sigmaX, None, sigmaY)
	result = np.copy(image)
	for _ in range(iterations):
		alpha = mask/255.
		result = alpha[:, :, None]*blurred_image + (1-alpha)[:, :, None]*result
		mask = cv2.GaussianBlur(mask, (mksize, mksize), msigmaX, None, msigmaY)
	return result

def softBlurRect(image, rect, ksize, sigmaX, *args, **kwargs):
	x,y,w,h = rect
	contours = [[np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]]
	return softBlurContours(image, contours, ksize, sigmaX, *args, **kwargs)

def blurContours(image, contours, ksize, sigmaX, *args):
	sigmaY = args[0] if len(args) > 0 else sigmaX
	mask = np.zeros(image.shape[:2])
	for i, contour in enumerate(contours):
		cv2.drawContours(mask, contour, i, 255, -1)
	blurred_image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX, None, sigmaY)
	result = np.copy(image)
	alpha = mask/255.
	result = alpha[:, :, None]*blurred_image + (1-alpha)[:, :, None]*result
	return result

def blurRect(image, rect, ksize, sigmaX, *args):
	x,y,w,h = rect
	contours = [[np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])]]
	return blurContours(image, contours, ksize, sigmaX, *args)