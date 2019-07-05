# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:07:57 2019

@author: shriram
"""

#importing modules
import numpy as np
from PIL import Image
from ISR.models import RDN

"""
  Initializes the model
  Parameters:
    weightPath: 'path to weights'
  Returns:
    rdn: 'model'
    
"""
def init(weightPath):
  rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
  rdn.model.load_weights(weightPath)
  return(rdn)
  
"""
  passes the image to the model to remove artifacts
  Parameters:
    img: 'image content'
    rdn: 'model'
  Returns:
    img_hr: 'The image content without Artifacts'
"""
def process(img,rdn):
  lr_img = np.array(img)
  sr_img = rdn.predict(lr_img)
  img_hr = Image.fromarray(sr_img)
  return(img_hr)
  
  """
  The image dimention is reduced into half of its orginal size and then process it
  Parameters: 
    imgPath: 'Path to image'
    rdn: 'model'
  Returns:
    img_hr: The image content without Artifacts
"""
def removeArtifacts(imgPath,rdn):
  img = Image.open(imgPath)
  size = img.width//2,img.height//2 
  img = img.resize(size)
  img_hr = process(img,rdn)
  return(img_hr)
  
#sample code for using the module
if __name__ == "__main__":
    weightPath = 'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
    rdn = init(weightPath)
    imgPath = "E:/IND0006216.jpg"
    img_hr = removeArtifacts(imgPath,rdn)