import cv2
import numpy as np
import torch
from ESRGAN import RRDBNet_arch as arch
import copy
from pyBlur import *
class Resizing:
	def __init__(self,modelPath):
		"""
	    Initializes the model
	    Parameters:
			modelPath: 'Path to Model'
        Architecture
	    """
	    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
	    model.load_state_dict(torch.load(modelPath), strict=True)
	    model.eval()
	    self.model = model

	def blurOut(self,image,portion):
	    """
	    Blurs the image outside the given portion
	    Parameters:
			image: contains image
			portion: contains the coordinates of the region not to blur [x1,y1,x2,y2]
	    Returns:
			image with blurred background
	    """
	    shape = image.shape
	    x1,y1,x2,y2 = portion
	    rect1 = [0,0,shape[1],y1]
	    rect2 = [0,y2,shape[1],shape[0]-y2]
	    rect3 = [0,y1,x1,y2-y1]
	    rect4 = [x2,y1,shape[1]-x2,y2-y1]
	    img1 = softBlurRect(image,rect1, 27, 9,9,55, iters = 7)
	    img2 = softBlurRect(img1,rect2, 27, 9,9,55, iters = 7) 
	    img3 = softBlurRect(img2,rect3, 27, 9,9,55, iters = 7) 
	    img4 = softBlurRect(img3,rect4, 27, 9,9,55, iters = 7)
	    return(img4)

	def superImpose(self,image,newImage):
	    """
	    Superimposes the newImage over the image
	    Parameters:
	    	image: background image to superimpose
	  		newImage: image to paste on the background image
	    Returns:
	  		returns required image
	    """
	    shape1 = image.shape
	    shape2 = newImage.shape
	    pip_h = int((shape1[0]-shape2[0])/2)
	    pip_w = int((shape1[1]-shape2[1])/2)
	    image[pip_h:pip_h+shape2[0],pip_w:pip_w+shape2[1]] = newImage
	    return(image,pip_w,pip_h)
	
	def upscale(self,img,model):
	    """
	    upscales the given img X 4
	    Parameters:
	  		img: contains the image to upscale
	    Returns:
	  		upscaled image
	    """
	    device = torch.device('cpu')
	    model = model.to(device)
	    img = img * 1.0 / 255
	    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
	    img_LR = img.unsqueeze(0)
	    img_LR = img_LR.to(device)
	    with torch.no_grad():
		output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
	    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
	    output = (output * 255.0).round()
	    return(output)

	def calculateDimention(self,height,width,aspectRatio):
		"""
		Calculates the new dimention
		Parameters:
			height: The height required
			width: The width required
		Returns:
			The new width and new height of image
		"""
		while(height > 0):
			newWidth = int(aspectRatio*height)
			newHeight = height
			if(newWidth <= width):
				break
			height = height -1
		  	return(newWidth,newHeight)

	def shrink(self,img,newHeight,newWidth,height,width):
		"""
		shrinks the image
		Parameters:
			img: 'The image to be resized'
			newWidth: 'new width'
			newHeight: 'new height'
		Returns:
			False if image does not fit properly
			The shrinked image if it fits
		"""
		img = cv2.resize(img,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
		background = cv2.resize(im,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
		image,stX,stY = self.superImpose(background,img)
		image = self.blurOut(image,[stX,stY,stX+img.shape[1],stY+img.shape[0]])
		diff = np.abs((width-newWidth)-(height-newHeight))
		if(diff > 600):
		print('Image Size Does not match')
		image = False
		return(image)

	def enlarge(self,im,newHeight,newWidth,height,width,model):
	    """
	    Enlarges the image
	    Parameters:
	    	newHeight: 'The new height to be resized'
	    	im: 'The image'
	    Returns:
	    	False if it is not upscaled
	    	image if it is upscaled
	    """
	    reqRatio = newHeight/im.shape[0]
	    n_x4 = reqRatio/4
		n_x4 = int(np.round(n_x4))
	    img = copy.deepcopy(im)
		for i in range(n_x4):
			print('upscaling')
			img = self.upscale(img,model)
	    if(n_x4 != 0):
			img = cv2.resize(img,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
	    background = cv2.resize(im,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
	    image,stX,stY = self.superImpose(background,img)
	    image = self.blurOut(image,[stX,stY,stX+img.shape[1],stY+img.shape[0]])
	    diff = np.abs((width-newWidth)-(height-newHeight))
	    if(diff > 600):
	    	image = False
	    	print('Image Size Does not match')
	    return(image)

	def resizing(self,imgPath,height,width,model):
		"""
		Resizes images according to the given size
		Parameters:
			imgPath: 'path to image'
			height: 'required height'
			width: 'required width'
		Returns
			False if image does not fit the size
			new image if it is resized
		"""
		flag = True
		newHeight, newWidth = 0,0
		im = cv2.imread(imgPath)
		shape = im.shape
		aspectRatio = shape[1]/shape[0]
		reqAspectRatio = width/height
		if(shape[0] == height and shape[1] == width):
			newWidth = width
			newHeight = height
		else:
			newWidth,newHeight = self.calculateDimention(height,width,aspectRatio)
		if(newHeight <= shape[0] and newWidth <= shape[1]):
		  	flag = self.shrink(im,newHeight,newWidth,height,width)
		  	return(flag)
		else:    
		  	flag =self.enlarge(im,newHeight,newWidth,height,width,model)
		  	return(flag)
	  
#sample code to run
if __name__=="main":
    imgName = 'E:/DemoImages/Resize/IND0011440.jpg'
    modelPath = 'C:/Users/shriram/Documents/resizing/ESRGAN/models/RRDB_ESRGAN_x4.pth'
    model = init(modelPath)
    newImage =  resizing(imgName,1500,1500,model)
    if(type(newImage)!= bool):
        cv2.imwrite('E:/resultjupyter.jpg',newImage)
        print('Done')
    else:
        print('Not possible')
