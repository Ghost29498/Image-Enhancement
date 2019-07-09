import numpy as np
import pandas as pd
import pickle
import os

from keras.models import load_model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class BlurDetection:
	def __init__(self,modelPath = 'none'):
		"""
		initializes the class
		Parameters:
			modelPath: 'path to model'
		"""
		self.goodLabel = 0
		self.badLabel = 1
		self.targetHeight = 192
		self.targetWidth = 192
		self.initializeModel()
		if(modelPath != 'none'):
			self.loadModel(modelPath)
	
	def initializeModel(self):
		"""
		initializes the model
		"""
		model = Sequential()

		# Layer 1
		numberOfFeaturesL1 = 32
		kernelSizeL1 = 5
		poolSizeL1 = 2
		actiavtionL1 = 'relu'
		model.add(Convolution2D(numberOfFeaturesL1, (kernelSizeL1,kernelSizeL1), input_shape=(self.targetHeight, self.targetWidth, 1)))
		model.add(Activation(actiavtionL1))
		model.add(MaxPooling2D(pool_size=(poolSizeL1, poolSizeL1)))

		# Layer 2
		numberOfFeaturesL2 = 64
		kernelSizeL2 = 5
		poolSizeL2 = 2
		activationL2 = 'relu'
		model.add(Convolution2D(numberOfFeaturesL2, (kernelSizeL2, kernelSizeL2))) 
		model.add(Activation(activationL2))
		model.add(MaxPooling2D(pool_size=(poolSizeL2, poolSizeL2)))    

		model.add(Flatten())

		# Layer 3
		numberOfFeaturesL3 = 1024
		activationL3 = 'relu'
		dropoutLevelL3 = 0.5
		model.add(Dense(numberOfFeaturesL3))
		model.add(Activation(activationL3))
		model.add(Dropout(dropoutLevelL3))

		# Layer 4
		numberOfFeaturesL4 = 512
		activationL4 = 'relu'
		dropoutLevelL4 = 0.5
		model.add(Dense(numberOfFeaturesL4))
		model.add(Activation(activationL4))
		model.add(Dropout(dropoutLevelL4))

		# Layer 5
		numberOfFeaturesL5 = 2
		activationL5 = 'softmax'
		model.add(Dense(numberOfFeaturesL5))
		model.add(Activation(activationL5))
		
		optimizer = 'adam'
		lossFunction = 'binary_crossentropy'
		model.compile(optimizer=optimizer, loss=lossFunction, metrics=['accuracy'])
		
		self.model = model
	
	def loadModel(self,modelPath):
		"""
		loads model from path
		Parameters:
			modelPath: 'path to model'
		"""
		self.model = load_model(modelPath)
	
	def loadImg(self,imagePath):
		"""
		loads images
		Parameters:
			imagePath: 'path to images'
		Returns:
			returns images
		"""
		# load image arrays
		X = []
		for filename in os.listdir(folderpath):
			if filename != '.DS_Store':
				imagepath = folderpath + filename
				img = image.load_img(imagepath, target_size=(self.targetHeight,self.targetWidth), color_mode = "grayscale")
				X.append(np.asarray(img))
			else:
				print(filename, 'not a pic')
		return(X)
	
	def createTrainDataSet(self,goodImg,badImg,train_size=0.91):
		"""
		creates train data set
		Parameters:
			goodImg: 'path to good images'
			badImg: 'Path to bad images'
			train_size: '% of images for training'
		Returns:
			returns training and test set
		"""
		goodImgList = self.loadImg(goodImg)
		badImgList = self.loadImg(badImg)
		X = goodImgList + badImgList
		y = [self.goodLabel]*len(goodImgList) + [self.badLabel]*len(badImgList)
		goodImgList = badImgList = 0
		X_stacked = np.stack(X)
		X_norm = X_stacked/255.0
		y_cat = to_categorical(y)
		X_norm = X_norm.reshape(X_norm.shape[0],self.targetHeight,self.targetWidth,1)
		X_train, X_test, y_train, y_test = train_test_split(X_norm, y_cat,stratify = y_cat ,train_size=train_size, random_state=42)
		X_train = X_train.reshape(X_train.shape[0], self.targetHeight,self.targetWidth, 1)
		X_test = X_test.reshape(X_test.shape[0], self.targetHeight,self.targetWidth, 1)
		X = y = X_stacked = X_norm = y_cat = 0
		return(X_train,X_test,y_train,y_test)
	
	def train(self,goodImg,badImg):
		"""
		trains the model
		Parameters:
			goodImg: path to good Image
			badImg: path to bad Image
		"""
		samplePerEpoch = 5000
		numberOfEpoch = 30
		numberOfValidation = 100
		X_train,X_test,y_train,y_test = self.createTrainDataSet(goodImg,badImg)
		dg = image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
		cb_es = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
		cb_mc = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
		self.model.fit_generator(dg.flow(X_train, y_train), samples_per_epoch=samplePerEpoch, nb_epoch=numberOfEpoch, validation_data=dg.flow(X_test, y_test), nb_val_samples=numberOfValidation, callbacks=[cb_es, cb_mc])		
	
	def test(self,imagePath):
		"""
		predicts the images in the given path
		Parameters:
			imagePath: 'path to image'
		Returns:
			predicted clasess
		"""
		X = self.loadImg(imagePath)
		testX = np.stack(X)/255.0
		textX = testX.reshape(textX.shape[0],self.targetHeight,self.targetWidth,1)
		return(self.model.predict_classes(testX))
	
	def saveModel(self,path):
		"""
		saves model in the given path
		Parameters:
			path: 'path to save model'
		"""
		self.model.save(path)
	
	def readImg(self,imagePath):
		"""
		reads image 
		Parameters:
			imagePath: 'path to image'
		Returns:
			image
		"""
		img = image.load_img(imagePath, target_size=(self.targetHeight,self.targetWidth), color_mode = "grayscale")
		return(img)
	
	def predict(self,img):
		"""
		predicts whether the image is blur or clear
		Parameters:
			img: 'path to image'
		Returns:
			returns prediction
		"""
		img = self.readImg(img)
		img = np.asarray(img)/255.0
		img = img.reshape(1,self.targetHeight,self.targetWidth,1)
		prediction = blurDetection.model.predict_classes(img)
		if(prediction[0] == self.goodLabel):
			return('good')
		else:
			return('bad')