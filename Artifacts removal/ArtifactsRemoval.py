# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:07:57 2019

@author: shriram
"""

#importing modules
import numpy as np
from PIL import Image
from ISR.models import RDN

class ArtifactsRemoval:
  def __init__(self,weightPath):
    """
    Initializes the model
    Parameters:
      weightPath: 'path to weights'
    Architecture Parameters:
       - D  - number of Residual Dense Blocks (RDB)
       - C  - number of convolutional layers stacked inside a RDB
       - G  - number of feature maps of each convolutional layers inside the RDBs 
       - G0 - number of feature maps for convolutions outside of RDBs and of each RBD output
       - x  - upscale parameter
    """
    rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
    rdn.model.load_weights(weightPath)
    self.rdn = rdn
  
  def process(self,img):
    """
    passes the image to the model to remove artifacts
    Parameters:
      img: 'image content'
    Returns:
      img_hr: 'The image content without Artifacts'
    """
    lr_img = np.array(img)
    sr_img = self.rdn.predict(lr_img)
    img_hr = Image.fromarray(sr_img)
    return(img_hr)
  def removeArtifacts(self,imgPath):
    """
    The image dimention is reduced into half of its orginal size and then process it
    Parameters: 
      imgPath: 'Path to image'
    Returns:
      img_hr: The image content without Artifacts
    """
    img = Image.open(imgPath)
    size = img.width//2,img.height//2 
    img = img.resize(size)
    img_hr = process(img)
    return(img_hr)
  
#sample code for using the module
if __name__ == "__main__":
    weightPath = 'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
    imgPath = "E:/IND0006216.jpg"
    artifactsRemoval = ArtifactsRemoval(weightPath)
    img_hr = artifactsRemoval.removeArtifacts(imgPath)
