import cv2
import numpy as np
import torch
from ESRGAN import RRDBNet_arch as arch
import copy
from pyBlur import *
"""
Initializes the model
    Parameters:
        modelPath: 'Path to Model'
    Returns:
        returns the model
"""
def init(modelPath):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(modelPath), strict=True)
    model.eval()
    return(model)
	
"""
Blurs the image outside the given portion
parameters:
  image: contains image
  portion: contains the coordinates of the region not to blur [x1,y1,x2,y2]
returns:
  image with blurred background
"""
def blurOut(image,portion):
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
	
"""
Superimposes the newImage over the image
parameters:
  image: background image to superimpose
  newImage: image to paste on the background image
returns:
  returns required image
"""
def superImpose(image,newImage):
    shape1 = image.shape
    shape2 = newImage.shape
    pip_h = int((shape1[0]-shape2[0])/2)
    pip_w = int((shape1[1]-shape2[1])/2)
    image[pip_h:pip_h+shape2[0],pip_w:pip_w+shape2[1]] = newImage
    return(image,pip_w,pip_h)

"""
upscales the given img X 4
parameters:
  img: contains the image to upscale
returns:
  upscaled image
"""
def upscale(img,model):
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

"""
  Calculates the new dimention
  Parameters:
    height: The height required
    width: The width required
  Returns:
    The new width and new height of image
"""
def calculateDimention(height,width,aspectRatio):
  while(height > 0):
          newWidth = int(aspectRatio*height)
          newHeight = height
          if(newWidth <= width):
              break
          height = height -1
  return(newWidth,newHeight)
  
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
def shrink(img,newHeight,newWidth,height,width):
  img = cv2.resize(img,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
  background = cv2.resize(im,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
  image,stX,stY = superImpose(background,img)
  image = blurOut(image,[stX,stY,stX+img.shape[1],stY+img.shape[0]])
  diff = np.abs((width-newWidth)-(height-newHeight))
  if(diff > 600):
    print('Image Size Does not match')
    image = False
  return(image)
  
"""
  Enlarges the image
  Parameters:
    newHeight: 'The new height to be resized'
    im: 'The image'
  Returns:
    False if it is not upscaled
    image if it is upscaled
"""
def enlarge(im,newHeight,newWidth,height,width,model):
    reqRatio = newHeight/im.shape[0]
    n_x4 = reqRatio/4
    if(n_x4-int(n_x4)>0.5):
      n_x4 = np.ceil(n_x4)
    else:
      n_x4 = np.floor(n_x4)
    i = 0
    img = copy.deepcopy(im)
    flag = False
    while(i<n_x4):
        print('upscaling')
        img = upscale(img,model)
        flag1 = True            
        i = i + 1
    if(flag):
        img = cv2.resize(img,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
    #print('Difference : ',np.abs((width-newWidth)-(height-newHeight)))
    background = cv2.resize(im,(newWidth,newHeight), interpolation = cv2.INTER_CUBIC)
    image,stX,stY = superImpose(background,img)
    image = blurOut(image,[stX,stY,stX+img.shape[1],stY+img.shape[0]])
    diff = np.abs((width-newWidth)-(height-newHeight))
    if(diff > 600):
      image = False
      print('Image Size Does not match')
    return(image)
	
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
def resizing(imgPath,height,width,model):
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
    newWidth,newHeight = calculateDimention(height,width,aspectRatio)
  if(newHeight <= shape[0] and newWidth <= shape[1]):
      #shrink
      flag = shrink(im,newHeight,newWidth,height,width)
      return(flag)
  else:    
      flag = enlarge(im,newHeight,newWidth,height,width,model)
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
