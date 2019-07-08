# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:14:20 2019

@author: ShriramSubramanian
"""

import numpy as np
import cv2
import torch
import sys
from util import *
import scipy.io as sio
sys.path.append('.\\Noise')
from Noise.models.DenoisingModels import *

class NoiseRemoval:
	def __init__(self,modelPath,device):
		"""
		initializes model
		Parameters:
			modelPath: 'path to model'
			device: 'device'
		Returns:
			returns the model
		"""
		self.device = torch.device(device)
		model = ntire_rdb_gd_rir_ver2().to(self.device)
		# load checkpoint of the model
		checkpoint = torch.load(modelPath,map_location=device)
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
		fixedHeight = 256
		fixedWidth = 256
		img = readImg(imgPath)
		img = resize(img,fixedHeight,fixedWidth)
		img = RGBToBGR(img)
		outimg = self.model.removeNoise(img)
		outimg = BGRToRGB(outimg)
		return(outimg)
   
if __name__=="__main__":
	modelPath = 'C:/Users/ShriramSubramanian/Noise Removal/NoiseRemoval/checkpoints/DGU-3DMlab1_track2.pkl'
	imgPath = 'G:/Print out/writeup/result (5).jpg'
	device = 'cpu'
	noiseRemoval = NoiseRemoval(modelPath,device)
	img = noiseRemoval.removeNoise(imgPath)
	cv2.imwrite('G:/Print out/writeup/asdfghjk123456.jpg',img)