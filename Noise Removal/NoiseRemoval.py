# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:14:20 2019

@author: ShriramSubramanian
"""

import numpy as np
import cv2
import torch
import sys
sys.path.append('.\\Noise')
from Noise.models.DenoisingModels import *
import scipy.io as sio

class NoiseRemoval:
    def __init__(self,modelPath):
        """
		initializes model
		Parameters:
			modelPath: 'path to model'
		Returns:
			returns the model
		"""
        self.device = torch.device('cpu')
        model = ntire_rdb_gd_rir_ver2().to(self.device)
		# load checkpoint of the model
        checkpoint = torch.load(modelPath,map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model
    def removeNoise(self,imgPath):
        """
        removes noise from images
			Parameters:
				imgPath: 'path to image'
			Returns:
				image without noise
		"""
        img = cv2.imread(imgPath)
        img = cv2.resize(img,(256,256))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outimg = self.model.removeNoise(img)
        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
        return(outimg)
    
if __name__=="__main__":
    modelPath = 'C:/Users/ShriramSubramanian/Noise Removal/NoiseRemoval/checkpoints/DGU-3DMlab1_track2.pkl'
    imgPath = 'G:/Print out/writeup/result (5).jpg'
    noiseRemoval = NoiseRemoval(modelPath)
    img = noiseRemoval.removeNoise(imgPath)
    cv2.imwrite('G:/Print out/writeup/asdfghjk123456.jpg',img)