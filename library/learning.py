# This script is for learning.

import cv2  
# cv2.setNumThreads(0)

import os
import shutil
import numpy as np
import random
import math


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utilities import resize_image, random_distort, load_train_data, load_train_data_multi, load_train_data_multi_pack
from networks_pytorch import create_nvidia_network_pytorch, ADDA_NVIDIA_FEATURE_CNN, ADDA_NVIDIA_REGRESSOR, ADDA_DOMAIN_DISCRIMINATOR, net_nvidia_pytorch_CNN, net_nvidia_pytorch_regressor
import time
import ntpath

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from torchviz import make_dot

import skimage.io
import matplotlib.pyplot as plt

from numpy.random import default_rng

from sys import platform
from PIL import Image

USE_KERAS = False
if USE_KERAS:
	import keras
	import tensorflow as tf
	from keras.utils.np_utils import to_categorical
	from keras.callbacks import ModelCheckpoint, CSVLogger
	from keras import backend as K
	# from keras.utils.vis_utils import plot_model
	from networks import net_lstm, create_nvidia_network, GAN_Nvidia, mean_accuracy



#tf.enable_eager_execution()



def gen_train_data_lstm_m2m(xList, yList, batchSize, nFramesSample):
	## get feature dimensions
	featureSample = resize_image(cv2.imread(xList[0]))
	
	## specify X and y shapes
	X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
	y = np.empty((batchSize, nFramesSample, 1))
	
	## generate training data
	sampleCount = 0
	while True:  
		for i in range(0, len(yList) - nFramesSample):
			## create a sample that has multiple frames
			for k in range(nFramesSample):
				X[sampleCount,k] = resize_image(cv2.imread(xList[i + k]))
				y[sampleCount,k] = yList[i + k]
				
			sampleCount += 1
				
			## yield a batch when #samples = batchSize
			if(sampleCount == batchSize):
				yield (X, y)
				X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
				y = np.empty((batchSize, nFramesSample, 1))
				sampleCount = 0
				
				
				
def gen_train_data_lstm_m2o(xList, yList, batchSize, nFramesSample):
	## get feature dimensions
	featureSample = resize_image(cv2.imread(xList[0]))
	
	## specify X and y shapes
	X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
	y = np.empty((batchSize, 1))
	
	## generate training data
	sampleCount = 0
	while True:  
		for i in range(0, len(yList) - nFramesSample):
			## create a sample that has multiple frames
			for k in range(nFramesSample):
				X[sampleCount,k] = resize_image(cv2.imread(xList[i + k]))
				
			y[sampleCount] = yList[i + nFramesSample]
			sampleCount += 1
				
			## yield a batch when #samples = batchSize
			if(sampleCount == batchSize):
				yield (X, y)
				X = np.empty((batchSize, nFramesSample, featureSample.shape[0], featureSample.shape[1], featureSample.shape[2]))
				y = np.empty((batchSize, 1))
				sampleCount = 0
					

def load_data_all(xList, yList):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	for i in range(len(yList)):
		image_path = xList[i]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")
		if not os.path.isfile(image_path):
			print(image_path)
		img = resize_image(cv2.imread(image_path))

		angle = yList[i]
		X.append(img)
		y.append(angle)

	return (np.array(X), np.array(y))

						
def gen_train_data_random(xList, yList, batchSize, fRandomDistort = False, fFlip = False, Maxup_flag = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])

	# For Maxup
	rng = default_rng()
	m=4 # TODO, currently fix m to 4
	perturbations = np.array([
							"trainB_R_darker_1", "trainB_R_darker_2", "trainB_R_darker_3", "trainB_R_darker_4", "trainB_R_darker_5", 
							"trainB_G_darker_1", "trainB_G_darker_2", "trainB_G_darker_3", "trainB_G_darker_4", "trainB_G_darker_5", 
							"trainB_B_darker_1", "trainB_B_darker_2", "trainB_B_darker_3", "trainB_B_darker_4", "trainB_B_darker_5", 
							"trainB_H_darker_1", "trainB_H_darker_2", "trainB_H_darker_3", "trainB_H_darker_4", "trainB_H_darker_5", 
							"trainB_S_darker_1", "trainB_S_darker_2", "trainB_S_darker_3", "trainB_S_darker_4", "trainB_S_darker_5", 
							"trainB_V_darker_1", "trainB_V_darker_2", "trainB_V_darker_3", "trainB_V_darker_4", "trainB_V_darker_5", 
							"trainB_R_lighter_1", "trainB_R_lighter_2", "trainB_R_lighter_3", "trainB_R_lighter_4", "trainB_R_lighter_5",
							"trainB_G_lighter_1", "trainB_G_lighter_2", "trainB_G_lighter_3", "trainB_G_lighter_4", "trainB_G_lighter_5",
							"trainB_B_lighter_1", "trainB_B_lighter_2", "trainB_B_lighter_3", "trainB_B_lighter_4", "trainB_B_lighter_5",
							"trainB_H_lighter_1", "trainB_H_lighter_2", "trainB_H_lighter_3", "trainB_H_lighter_4", "trainB_H_lighter_5",
							"trainB_S_lighter_1", "trainB_S_lighter_2", "trainB_S_lighter_3", "trainB_S_lighter_4", "trainB_S_lighter_5",
							"trainB_V_lighter_1", "trainB_V_lighter_2", "trainB_V_lighter_3", "trainB_V_lighter_4", "trainB_V_lighter_5",
							"trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5",
							"trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5",
							"trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5"])
	tot_p = len(perturbations)

	while True:	   
		for i in range(len(yList)):
			image_path = xList[i]
			angle = yList[i]

			if Maxup_flag:
				#idx = random.randint(tot_p, size=(m))
				idx = rng.choice(tot_p, size=m, replace=False)
				#idx = [0]
				for j in idx:
					new_train_folder = perturbations[j]
					new_image_path = image_path.replace("trainB", new_train_folder)
					#print(new_image_path)
					img = resize_image(cv2.imread(new_image_path))
					X.append(img)
					y.append(angle)


			else:
				if not os.path.isfile(image_path):
					image_path = image_path.replace(".jpg", "_fake.png")
				if not os.path.isfile(image_path):
					print(image_path)
				img = resize_image(cv2.imread(image_path))


				#TODO, tmp shift code
				# if "trainC1" in image_path:
				# 	B_means = (np.array([-0.05958, 0.03200, -0.06430]) + 1 ) * 127.5
				# 	B_stds = np.array([0.52423, 0.11046, 0.13780]) * 127.5
				# 	C1_means = (np.array([-0.26482, 0.08651, -0.05964]) + 1 ) * 127.5
				# 	C1_stds = np.array([0.47422, 0.10676, 0.10106]) * 127.5
				# 	for cn in range(3):
				# 		#img[:,:,cn] = (((img[:,:,cn] / 127.5 - 1) - C1_means[cn]) / C1_stds[cn] * B_stds[cn] + B_means[cn] + 1) * 127.5
				# 		#img[:,:,cn] = img[:,:,cn] * (1 / 127.5 / C1_stds[cn] * B_stds[cn] * 127.5) + ((-1 - C1_means[cn]) / C1_stds[cn] * B_stds[cn] + B_means[cn] + 1) * 127.5
				# 		#img[:,:,cn] = (img[:,:,cn]- C1_means[cn]) / C1_stds[cn] * B_stds[cn] + B_means[cn]
				# 		img[:,:,cn] = img[:,:,cn] * (B_stds[cn] / C1_stds[cn]) + ((- C1_means[cn]) / C1_stds[cn] * B_stds[cn] + B_means[cn])

				if fRandomDistort:
					print('######### Applying random distortion #########')
					img, angle = random_distort(img, angle)
				X.append(img)
				y.append(angle)

			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) >= batchSize:
				# yield (np.array(X), np.array(y))
				yield (np.array(X), np.array(y), 0)
				X, y = ([],[])
				xList, yList = shuffle(xList, yList)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)

def gen_train_data_random_pack_channel(xList, yList, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	while True:	   
		for i in range(len(yList)):
			for j in range(len(xList[i])):
				image_path = xList[i][j]
				if not os.path.isfile(image_path):
					image_path = image_path.replace(".jpg", "_fake.png")
				img_1 = resize_image(cv2.imread(image_path))
				if j == 0:
					img = img_1
				else:
					img = np.concatenate((img, img_1), axis=2)

			#noise = np.random.uniform(low=0, high=255, size=(img.shape[0], img.shape[1], 1))
			#img = np.concatenate((img, noise), axis=2)

			angle = yList[i][0]
			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)


			X.append(img)
			y.append(angle)
			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				yield (np.array(X), np.array(y))
				X, y = ([],[])
				xList, yList = shuffle(xList, yList)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)

def gen_train_data_random_AdvProp(xList, yList, xList_advp, yList_advp, batchSize, fRandomDistort = False, fFlip = False):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	X_advp, y_advp = ([],[])
	while True:	   
		for i in range(max(len(yList), len(yList_advp))):
			image_path = xList[i%len(xList)]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			angle = yList[i%len(yList)]

			image_path_advp = xList_advp[i%len(xList_advp)]
			if not os.path.isfile(image_path_advp):
				image_path_advp = image_path_advp.replace(".jpg", "_fake.png")
			img_advp = resize_image(cv2.imread(image_path_advp))
			angle_advp = yList_advp[i%len(yList_advp)]

			if fRandomDistort:
				print('######### Applying random distortion #########')
				img, angle = random_distort(img, angle)
				img_advp, angle_advp = random_distort(img_advp, angle_advp)


			#X.append([img, img_advp])
			#y.append([angle, angle_advp])
			X.append(img)
			y.append(angle)
			X_advp.append(img_advp)
			y_advp.append(angle_advp)

			
			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				#yield (np.array(X), np.array(y))
				#yield (np.array([X, X_advp]), np.array([y, y_advp]))
				yield [np.array(X), np.array(X_advp)], [np.array(y), np.array(y_advp)]
				X, y = ([],[])
				X_advp, y_advp = ([],[])
				xList, yList = shuffle(xList, yList)
				xList_advp, yList_advp = shuffle(xList_advp, yList_advp)
				
			## flip an image horizontally along its corresponding steering angle
			if fFlip:
				angleThreshold = 0.33
				if abs(angle) > angleThreshold:
					X.append(cv2.flip(img, 1))
					y.append(angle * -1)
					if len(X) == batchSize:
						yield (np.array(X), np.array(y))
						X, y = ([],[])
						xList, yList = shuffle(xList, yList)


def gen_train_data_random_featshift(xList, yList, xList_advp, yList_advp, batchSize):
	xList, yList = shuffle(xList, yList)
	X,y = ([],[])
	X_advp, y_advp = ([],[])
	while True:	   
		for i in range(max(len(yList), len(yList_advp))):
			image_path = xList[i%len(xList)]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			angle = yList[i%len(yList)]

			image_path_advp = xList_advp[i%len(xList_advp)]
			if not os.path.isfile(image_path_advp):
				image_path_advp = image_path_advp.replace(".jpg", "_fake.png")
			# img_advp = resize_image(cv2.imread(image_path_advp))
			feat_path_advp = image_path_advp.replace('trainC1', 'trainC1_feat').replace('.jpg', '.npy')
			feature = np.load(feat_path_advp)
			angle_advp = yList_advp[i%len(yList_advp)]

			#X.append([img, img_advp])
			#y.append([angle, angle_advp])
			X.append(img)
			y.append(angle)
			X_advp.append(feature)
			y_advp.append(angle_advp)

			## when a batch is ready, yield, and prepare for the next batch
			if len(X) == batchSize:
				#yield (np.array(X), np.array(y))
				#yield (np.array([X, X_advp]), np.array([y, y_advp]))
				yield [np.array(X), np.array(X_advp)], [np.array(y), np.array(y_advp)]
				X, y = ([],[])
				X_advp, y_advp = ([],[])
				xList, yList = shuffle(xList, yList)
				xList_advp, yList_advp = shuffle(xList_advp, yList_advp)
				
class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, labels = sample

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		#image = image.transpose((2, 0, 1))
		return (torch.Tensor(image), torch.Tensor(labels))

class DrivingDataset_pytorch(torch.utils.data.Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, xTrainList, yTrainList, transform=None, withFFT=False, Maxup_flag=False):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.xTrainList = xTrainList
		self.yTrainList = yTrainList
		self.transform = transform
		self.withFFT = withFFT
		self.Maxup_flag = Maxup_flag
		self.rng = default_rng()

	def __len__(self):
		return len(self.yTrainList)

	def __getitem__(self, idx):
		img_path = self.xTrainList[idx]

		img_paths = img_path
		if not isinstance(img_path, (list, tuple, np.ndarray)):
			img_paths = [img_path]

		if self.Maxup_flag:
			m=4

			perturbations = np.array([
							 "trainB_R_darker_1", "trainB_R_darker_2", "trainB_R_darker_3", "trainB_R_darker_4", "trainB_R_darker_5", 
							 "trainB_G_darker_1", "trainB_G_darker_2", "trainB_G_darker_3", "trainB_G_darker_4", "trainB_G_darker_5", 
							 "trainB_B_darker_1", "trainB_B_darker_2", "trainB_B_darker_3", "trainB_B_darker_4", "trainB_B_darker_5", 
							 "trainB_H_darker_1", "trainB_H_darker_2", "trainB_H_darker_3", "trainB_H_darker_4", "trainB_H_darker_5", 
							 "trainB_S_darker_1", "trainB_S_darker_2", "trainB_S_darker_3", "trainB_S_darker_4", "trainB_S_darker_5", 
							 "trainB_V_darker_1", "trainB_V_darker_2", "trainB_V_darker_3", "trainB_V_darker_4", "trainB_V_darker_5", 
							 "trainB_R_lighter_1", "trainB_R_lighter_2", "trainB_R_lighter_3", "trainB_R_lighter_4", "trainB_R_lighter_5",
							 "trainB_G_lighter_1", "trainB_G_lighter_2", "trainB_G_lighter_3", "trainB_G_lighter_4", "trainB_G_lighter_5",
							 "trainB_B_lighter_1", "trainB_B_lighter_2", "trainB_B_lighter_3", "trainB_B_lighter_4", "trainB_B_lighter_5",
							 "trainB_H_lighter_1", "trainB_H_lighter_2", "trainB_H_lighter_3", "trainB_H_lighter_4", "trainB_H_lighter_5",
							 "trainB_S_lighter_1", "trainB_S_lighter_2", "trainB_S_lighter_3", "trainB_S_lighter_4", "trainB_S_lighter_5",
							 "trainB_V_lighter_1", "trainB_V_lighter_2", "trainB_V_lighter_3", "trainB_V_lighter_4", "trainB_V_lighter_5",
							 "trainB_blur_1", "trainB_blur_2", "trainB_blur_3", "trainB_blur_4", "trainB_blur_5",
							 "trainB_noise_1", "trainB_noise_2", "trainB_noise_3", "trainB_noise_4", "trainB_noise_5",
							 "trainB_distort_1", "trainB_distort_2", "trainB_distort_3", "trainB_distort_4", "trainB_distort_5"])
			tot_p = len(perturbations)
			# pidx = random.randint(0,tot_p)
			pidx = self.rng.choice(m, size=m, replace=False)
			# pidx=[0,0,0,0]
			img_paths = []
			for j in pidx:
				new_train_folder = perturbations[j]
				img_path1 = img_path.replace("trainB", new_train_folder)
				img_paths.append(img_path1)

		# gt_value = np.random.rand()
		for i in range(len(img_paths)):
			img_path = img_paths[i]
			if not os.path.isfile(img_path):
				print(img_path, " not exists")

			img_1 = Image.open(img_path)
			img_1 = img_1.convert("RGB")
			# img_1.show()

			if self.withFFT:
				img_2 = img_1.resize((200,66))
				img_2 = np.array(img_2)
				img_2 = np.fft.fft2(img_2, axes=[0,1])
				img_2 = np.fft.fftshift(img_2)
				img_2 = img_2/np.mean(np.abs(img_2)) # normalize mean to 1
				img_2 = np.concatenate((img_2.real, img_2.imag),axis=2)

			if self.transform:
				img_1 = self.transform(img_1)
				if self.withFFT:
					img_2 = transforms.ToTensor()(img_2).float()
					img_1 = torch.cat((img_1, img_2), 0)

			if i == 0:
				img = img_1
			else:
				img = torch.cat((img, img_1), 0)

			################################ additional channels #####################################
			# random values
			# random_channel = torch.rand(img.shape)
			# img = torch.cat((img, random_channel), 0)

			# ground truth values
			# gt_channel = torch.tensor(np.full(img.shape, gt_value), dtype=torch.float32)
			# img = torch.cat((img, gt_channel), 0)

			# mask generation
			# mask_channel = cv2.imread("mask_ori.png")
			# mask_channel = cv2.resize(mask_channel, (200,66))
			# for i in range(33, 66):
			#	 for j in range(200):
			#		 mask_channel[i,j] = mask_channel[i,j]*(1-(i-33)*0.015)
			# cv2.imwrite("mask.png", mask_channel)

			# mask
			# img_1 = Image.open("mask.png")
			# img_1 = img_1.convert("RGB")

			# if self.transform:
			#	 img_1 = self.transform(img_1)

			# img = torch.cat((img, img_1), 0)


			# mask_channel = cv2.imread("mask.png")

			# cv2.imshow("img", mask_channel)
			# cv2.waitKey(0)
			###################################################################################################

		label = self.yTrainList[idx]

		# label = gt_value # TODO

		if isinstance(label, (list, tuple, np.ndarray)):
			label = label[0]
		label = np.array([label])
		label = torch.tensor(label).float()
		return img, label, img_path
		# return img, label

	# def __getitem__(self, idx): # for augmix
	#	 if torch.is_tensor(idx):
	#		 idx = idx.tolist()

	#	 img_name = self.xTrainList[idx]
	#	 image = cv2.imread(img_name)

	#	 image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
	#	 image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

	#	 image = np.transpose(image, (2, 0, 1))
	#	 image = image.astype('float32')

	#	 labels = self.yTrainList[idx]

	#	 # image = np.zeros((200,66,3))
	#	 # labels = 0.

	#	 labels = np.array([labels])
	#	 labels = labels.astype('float32')

	#	 sample = (image, labels)

	#	 # if self.transform:
	#	 #	 sample = self.transform(sample)

	#	 return sample


'''
def train_dnn(imageDir, labelPath, outputPath, netType, flags, specs):
	
	## assigning variables
	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
	batchSize 	   = specs[0]
	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	## prepare the data
	xList, yList = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)
	
	## change the data format if necessary
	if fClassifier:
		print('\n######### Classification #########')
		yTrainList = to_categorical(yTrainList, num_classes = nClass)
		yValidList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
	
	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = net_nvidia(fClassifier, nClass)
		trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
		validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2o(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2o(xValidList, yValidList, batchSize, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)
		trainGenerator = gen_train_data_lstm_m2m(xTrainList, yTrainList, batchSize, nFramesSample)
		validGenerator = gen_train_data_lstm_m2m(xValidList, yValidList, batchSize, nFramesSample)

	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	else:
		shutil.rmtree(outputPath)
		os.makedirs(outputPath)
	modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
	lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
	
	## train
	nTrainStep = int(len(yTrainList)/batchSize) + 1
	nValidStep = int(len(yValidList)/batchSize) + 1
	net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
	verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)
	net.save(outputPath + 'model-final.h5')
	print(net.summary())
'''

def normalize_value(value_list):
	THRESH_HOLDS = 24
	value_list = np.clip(value_list, -THRESH_HOLDS, THRESH_HOLDS)
	value_list = value_list + THRESH_HOLDS + 0.5
	return value_list

def get_mean_std():

	mean2 = [-0.08302036,-0.20238686,0.4779597,-0.001428917,-1.0989053,0.11795678,0.4835723,0.17353112,\
			-0.4888867,-0.057860363,-0.19815631,0.54467463,0.48134643,0.20496474,-0.31805158,-0.5533684,\
			0.38188627,-0.4976213,-0.13837108,-0.899611,-0.47604445,0.20196259,-0.59547615,0.14284715,\
			-0.16108884,0.29046857,-1.0335027,0.015435962,-0.55076504,0.06926734,-0.12838832,0.2616188,\
			0.540464,-0.17048602,-0.10068798,-0.48923278,-0.057503913,-0.051794812,0.47316852,0.1445276,\
			-0.020938095,-0.23620464,-0.016980559,0.2566837,-0.038310815,-0.038774405,-1.8071651,-0.115686394,\
			-0.7175674,-0.97459745,-0.38946486,-0.3183761,-0.53616697,-0.21956977,0.41411576,-0.30640075,\
			-0.45674223,-0.020145502,0.20598692,0.46789482,-0.5337459,-1.4274683,-0.05283879,-1.0345378]

	std2 = [2.0937233,2.2896929,3.5013025,1.9630629,2.5071824,1.6628706,1.7510566,2.425439,\
			2.0053973,1.4680599,1.855911,1.8895342,1.5442249,1.9109975,1.53864,2.2669528,\
			2.3621962,2.3518426,2.1757283,2.2138484,1.9544407,1.2151252,2.1721957,1.4883112,\
			1.953139,1.8143281,2.2361205,1.7194833,2.016445,1.8745058,1.3707753,1.4898472,\
			1.9244006,1.7030011,0.8539401,1.7533897,1.708709,1.6200079,2.2226865,1.6120292,\
			1.8885119,1.6399024,1.6098967,1.4447424,2.6011627,1.871935,3.2005286,2.4882145,\
			2.1371593,3.0658402,1.9056149,2.107141,2.0150385,2.0149024,2.0458705,1.8512557,\
			1.9597613,1.4950666,2.492967,2.2759092,2.3835964,3.5994263,2.2815502,2.1653445]

	mean2_new = []
	for v in range(len(mean2)):
		mean_ext = np.full((128,1,18), mean2[v])
		mean2_new.append(mean_ext)
	mean2 = np.array(mean2_new).transpose((1,0,2,3))

	std2_new = []
	for v in range(len(std2)):
		std_ext = np.full((128,1,18), std2[v])
		std2_new.append(std_ext)
	std2 = np.array(std2_new).transpose((1,0,2,3))

	return torch.Tensor(mean2).cuda(), torch.Tensor(std2).cuda()

def train_dnn_multi(imageDir_list, labelPath_list, outputPath, netType, flags, specs, modelPath = "", 
	trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imageDir_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, 
	pack_flag=False, mid=0, Maxup_flag=False, pytorch_flag=False, lr=0.0001, window_size_lstm=16, withFFT=False):
	
	startall = time.time()

	## assigning variables
	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
	batchSize 	   = specs[0]
	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]

	specialFilter = 0
	if BN_flag == 7:
		specialFilter = 2
	
	## prepare the data
	#xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio, specialFilter=True)
	if not pack_flag:
		xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio, specialFilter=specialFilter)
	else:
		xList, yList = load_train_data_multi_pack(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio)

	if BN_flag == 7: # LSTM
		valid_size=0.1
		split_pos = int(len(xList)*(1-valid_size))
		# print(split_pos)
		xTrainList = np.array(xList[0:split_pos])
		xValidList = np.array(xList[split_pos:])

		yTrainList = np.array(yList[0:split_pos])
		yValidList = np.array(yList[split_pos:])
	else:
		xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
		yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)


	#TODO
	# imageDir_list_valid=[imageDir_list[0].replace("trainWaymo", "valWaymo")]
	# labelPath_list_valid=[labelPath_list[0].replace("train", "val")]
	# # imageDir_list_valid=['/scratch/yushen/SAAP_Auto-driving_Platform/Data/udacityA_nvidiaB/valWaymo/']
	# # labelPath_list_valid=['/scratch/yushen/SAAP_Auto-driving_Platform/Data/udacityA_nvidiaB/labelsWaymo_val.csv']
	# xValidList, yValidList = load_train_data_multi(imageDir_list_valid, labelPath_list_valid, nRep, fThreeCameras, trainRatio, specialFilter=specialFilter)
	# xValidList = np.array(xValidList)
	# yValidList = np.array(yValidList)


	# yTrainList/=15
	# yValidList/=15

	if (BN_flag == 2) or (BN_flag == 3):
		xList_advp, yList_advp = load_train_data_multi(imageDir_list_advp, labelPath_list_advp, nRep, fThreeCameras, trainRatio_advp, specialFilter=specialFilter)

		xTrainList_advp, xValidList_advp = train_test_split(np.array(xList_advp), test_size=0.1, random_state=42)
		yTrainList_advp, yValidList_advp = train_test_split(np.array(yList_advp), test_size=0.1, random_state=42)
	
	## change the data format if necessary
	if fClassifier:
		print('\n######### Classification #########')
		yTrainList = normalize_value(yTrainList)
		yTrainList = to_categorical(yTrainList, num_classes = nClass)
		yValidList = normalize_value(yValidList)
		yValidList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
	
	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o, 5: GAN
	if pytorch_flag:
		nChannel = 3
		if pack_flag:
			nChannel = nChannel*len(imageDir_list)
		#net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
		net = create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel, withFFT=withFFT)
		if (BN_flag == 3):
			trainGenerator = gen_train_data_random_featshift(xTrainList, yTrainList, xTrainList_advp, yTrainList_advp, batchSize)
		else:

			size = (66, 200)
			if BN_flag == 8 or BN_flag == 10: #resnet
				size = (64, 64)
			transform=transforms.Compose([
				# transforms.Resize((66, 200)),
				transforms.Resize(size),
				transforms.ToTensor(),
				# transforms.Lambda(lambda rgb_img: 
				# 	torch.matmul(rgb_img.permute(1, 2, 0), 
				# 	yuv_weight.transpose(0, 1)).permute(2, 0, 1)),
				])

			train_dataset = DrivingDataset_pytorch(xTrainList, yTrainList, transform=transform, withFFT=withFFT, Maxup_flag=Maxup_flag)
			valid_dataset = DrivingDataset_pytorch(xValidList, yValidList, transform=transform, withFFT=withFFT)
			#dataset = DrivingDataset_pytorch(xTrainList, yTrainList)

			shuffle_flag = True
			if BN_flag == 7:
				shuffle_flag = False

			if platform == "win32":
				trainGenerator = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle_flag, num_workers=0)
				validGenerator = DataLoader(valid_dataset, batch_size=batchSize, shuffle=shuffle_flag, num_workers=0)
				#validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
			else:
				trainGenerator = DataLoader(train_dataset, batch_size=batchSize, shuffle=shuffle_flag, num_workers=8, pin_memory=True)
				validGenerator = DataLoader(valid_dataset, batch_size=batchSize, shuffle=shuffle_flag, num_workers=8, pin_memory=True)

			# if Maxup_flag:
			# 	trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize, Maxup_flag=Maxup_flag)
			# 	validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)

		print(net)
	else:
		if netType == 1:
	# 		outputPath = trainPath + 'trainedModels/models-cnn/';
			nChannel = 3
			if pack_flag:
				nChannel = 3*len(imageDir_list)

			if BN_flag == 3:
				net, netI, netF = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
			else:
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)

			if BN_flag <= 1 or BN_flag == 5:
				if not pack_flag:
					trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize, Maxup_flag=Maxup_flag)
					validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
				else:
					trainGenerator = gen_train_data_random_pack_channel(xTrainList, yTrainList, batchSize)
					validGenerator = gen_train_data_random_pack_channel(xValidList, yValidList, batchSize)
			elif (BN_flag == 2) or (BN_flag == 3):
				trainGenerator = gen_train_data_random_AdvProp(xTrainList, yTrainList, xTrainList_advp, yTrainList_advp, batchSize)
				validGenerator = gen_train_data_random_AdvProp(xValidList, yValidList, xValidList_advp, yValidList_advp, batchSize)

		elif netType == 2:
	# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
			net = net_lstm(2, nFramesSample)
			trainGenerator = gen_train_data_lstm_m2o(xTrainList, yTrainList, batchSize, nFramesSample)
			validGenerator = gen_train_data_lstm_m2o(xValidList, yValidList, batchSize, nFramesSample)
		elif netType == 3:
	# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
			net = net_lstm(3, nFramesSample)
			trainGenerator = gen_train_data_lstm_m2m(xTrainList, yTrainList, batchSize, nFramesSample)
			validGenerator = gen_train_data_lstm_m2m(xValidList, yValidList, batchSize, nFramesSample)
		elif netType == 5:
			net = GAN_Nvidia()
			trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
			validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)

		print(net.summary())


	if modelPath != "":
		print("pretrain modelPath: ", modelPath)

		#print(net.summary())
		#print(net.layers[2].get_weights())
		#print(net.layers[13].get_weights())

		if pytorch_flag:
			net.load_state_dict(torch.load(modelPath))
		else:
			if (BN_flag == 3):
				netF.load_weights(modelPath)
				for id in range(len(netF.layers)):
					netF.layers[id].trainable = False

				net_untrain = create_nvidia_network(0, fClassifier, nClass)
				for id in range(len(net_untrain.layers)):
					netI.layers[id].set_weights(net_untrain.layers[id].get_weights())
					netI.layers[id].trainable = True

				# print('refreeze')
				# for layer in net.layers:
				# 	print(layer.trainable)
			else:
				net.load_weights(modelPath)

		# net1 = create_nvidia_network(0, fClassifier, nClass, nChannel)
		# print(net1.summary())
		# net1.load_weights(modelPath)
		# print(net1.layers[13].get_weights())

		# #print(net.get_weights())
		# #print(net.layers[2].get_weights())
		# #print(net.layers[13].get_weights())
		# print(net.layers[19].get_weights())

		#print(partialPreModel)
		if partialPreModel:
			print("partial PreModel activate")
			#net_untrain = net_nvidia(fClassifier, nClass)
			start_layer_id=8
			for i in range(start_layer_id):
				net.layers[i].trainable = False
			#for i in range(start_layer_id, len(net.layers)):
				#net.layers[i].set_weights(net_untrain.layers[i].get_weights())
			#	net.layers[i].trainable = False
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
		if reinitHeader:
			print("reinit header activate")
			net_untrain = create_nvidia_network(BN_flag, fClassifier, nClass)
			net.layers[-1].set_weights(net_untrain.layers[-1].get_weights())
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
		if reinitBN:
			net_untrain = create_nvidia_network(BN_flag, fClassifier, nClass)
			BN_layer_ids = [3, 6, 9, 12, 15, 19, 22, 25]
			for id in BN_layer_ids:
				net.layers[id].set_weights(net_untrain.layers[id].get_weights())
			net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=[mean_accuracy])
			#net.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	#else:
	#	shutil.rmtree(outputPath)
	#	os.makedirs(outputPath)

	## train
	if pytorch_flag:
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		net = net.cuda()
		criterion = nn.MSELoss()
		#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		#optimizer = optim.Adam(net.parameters(), lr=0.0001)
		optimizer = optim.Adam(net.parameters(), lr=lr)
		train_batch_num = int(len(yTrainList)/batchSize)
		valid_batch_num = int(len(yValidList)/batchSize)
		thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
		# thresh_holds = [1.5, 3, 7.5, 15, 30, 75]

		#TODO, for shift
		mean2, std2 = get_mean_std()

		f_log = open(outputPath + 'loss-log', 'w')
		f_log.write('epoch,loss,mean_accuracy,val_loss,val_mean_accuracy\n')
		
		last_time = startall
		timing_list=[]
		for epoch in range(nEpoch):  # loop over the dataset multiple times
			start = time.time()
			running_loss = 0.0
			train_acc_list = [0,0,0,0,0,0]
			data_number = 0
			net.train()

			for i, (inputs, labels,_) in enumerate(trainGenerator):
				# inputs.requires_grad = True  #TODO
				# labels = labels.numpy().flatten()
				if BN_flag == 3:
					image, feature = inputs
					labels1, labels2 = labels
					# image = np.transpose(image, (0, 3, 1, 2))
					inputs = [image, feature]
					labels = np.concatenate((labels1, labels2))
					outputs = net(torch.Tensor(image).cuda(), torch.Tensor(feature).cuda(), mean2, std2)
					data_number += image.shape[0]
				elif BN_flag == 7:
					if len(labels) < batchSize:
						break
					current_pos = random.randrange(window_size_lstm)
					inputs_seq = []
					labels_seq = []
					while current_pos + window_size_lstm <= batchSize:
						inputs_seq.append(inputs[current_pos:current_pos + window_size_lstm])
						labels_seq.append(labels[current_pos:current_pos + window_size_lstm])
						current_pos = current_pos + window_size_lstm

					inputs_seq = torch.stack(inputs_seq)
					labels = torch.stack(labels_seq)
					outputs,_ = net(torch.Tensor(inputs_seq).cuda(non_blocking=True))
					data_number += inputs_seq.shape[0]*inputs_seq.shape[1]
				else:
					# if Maxup_flag:
					# 	inputs = np.transpose(inputs, (0, 3, 1, 2))
					if Maxup_flag: # new pytorch version
						m=4
						inputs = inputs.reshape(inputs.shape[0]*m, int(inputs.shape[1]/m), inputs.shape[2], inputs.shape[3])
						
					outputs,_ = net(torch.Tensor(inputs).cuda(non_blocking=True))
					data_number += inputs.shape[0]

				# make_dot(outputs).render("rnn_torchviz", format="png")

				# backward + optimize
				if type(labels) is not np.ndarray:
					labels = labels.numpy() # new pytorch version

				if Maxup_flag:# and epoch >= 5:
					m=4
					# old version
					# prediction = outputs.cpu().detach().numpy().flatten().reshape((int(batchSize/m), m))
					# labels = labels.reshape((int(batchSize/m), m))

					# new version
					prediction = outputs.cpu().detach().numpy().flatten().reshape(batchSize, m)
					labels = labels.flatten().repeat(m).reshape(batchSize, m)

					error = np.abs(prediction - labels)

					max_ids = np.argmax(error, axis=1)
					b = np.zeros((max_ids.size, m))
					b[np.arange(max_ids.size),max_ids] = True
					labels = np.where(b < 0.5, prediction, labels)

				# labels = labels.flatten()
				# labels_2d = labels.reshape((labels.shape[0], 1))
				labels_2d = labels.reshape(outputs.shape)

				# zero the parameter gradients
				optimizer.zero_grad()

				loss = criterion(outputs, torch.Tensor(labels_2d).cuda(non_blocking=True))

				loss.backward()

				# print(inputs.grad)		   #TODO
				# print(inputs.grad.shape)

				# print('=====================================================================================')
				# input_grad_np = inputs.grad.cpu().detach().numpy()
				# print(np.mean(np.abs(input_grad_np[:,0,:,:])))
				# print(np.mean(np.abs(input_grad_np[:,1,:,:])))
				# print(np.mean(np.abs(input_grad_np[:,2,:,:])))
				# print('------------------------------------------------------------------')
				# print(np.mean(np.abs(input_grad_np[:,3,:,:])))
				# print(np.mean(np.abs(input_grad_np[:,4,:,:])))
				# print(np.mean(np.abs(input_grad_np[:,5,:,:])))
				# adf


				optimizer.step()

				# print statistics
				running_loss += loss.item()

				prediction_error = np.abs(outputs.cpu().detach().numpy().flatten()-labels.flatten())
				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error < thresh_hold)
					train_acc_list[j] += acc_count

				if i >= train_batch_num-1:
					break

			for j in range(len(train_acc_list)):
				train_acc_list[j] = train_acc_list[j] / data_number
			train_acc = np.mean(train_acc_list)


			valid_loss = 0.0
			valid_acc_list = [0,0,0,0,0,0]
			data_number = 0
			net.eval()

			for i, (inputs, labels, _) in enumerate(validGenerator):
				# labels = labels.numpy().flatten()
				if BN_flag == 3:
					image, feature = inputs
					labels1, labels2 = labels
					# image = np.transpose(image, (0, 3, 1, 2))
					inputs = [image, feature]
					labels = np.concatenate((labels1, labels2))
					outputs = net(torch.Tensor(image).cuda(), torch.Tensor(feature).cuda(), mean2, std2)
					data_number += image.shape[0]
				elif BN_flag == 7:
					current_pos = random.randrange(window_size_lstm)
					inputs_seq = []
					labels_seq = []
					while current_pos + window_size_lstm <= batchSize:
						inputs_seq.append(inputs[current_pos:current_pos + window_size_lstm])
						labels_seq.append(labels[current_pos:current_pos + window_size_lstm])
						current_pos = current_pos + window_size_lstm

					inputs_seq = torch.stack(inputs_seq)
					labels = torch.stack(labels_seq)
					outputs,_ = net(torch.Tensor(inputs_seq).cuda(non_blocking=True))
					data_number += inputs_seq.shape[0]*inputs_seq.shape[1]
				else:
					# if Maxup_flag:
					# 	inputs = np.transpose(inputs, (0, 3, 1, 2))
					outputs,_ = net(torch.Tensor(inputs).cuda(non_blocking=True))
					data_number += inputs.shape[0]

				# labels = labels.flatten()
				# labels_2d = labels.reshape((labels.shape[0], 1))
				labels_2d = labels.reshape(outputs.shape)

				# zero the parameter gradients
				optimizer.zero_grad()

				loss = criterion(outputs, torch.Tensor(labels_2d).cuda(non_blocking=True))

				# print statistics
				valid_loss += loss.item()

				if type(labels) is not np.ndarray:
					labels = labels.numpy() # new pytorch version

				prediction_error = np.abs(outputs.cpu().detach().numpy().flatten()-labels.flatten())

				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error < thresh_hold)
					valid_acc_list[j] += acc_count

				if i >= valid_batch_num-1:
					break

			for j in range(len(valid_acc_list)):
				valid_acc_list[j] = valid_acc_list[j] / data_number
			val_acc = np.mean(valid_acc_list)



			# (inputs, labels, _) = load_data_all(xValidList, yValidList)
			# inputs = np.transpose(inputs, (0, 3, 1, 2))
			# labels_2d = labels.reshape((labels.shape[0], 1))

			# if BN_flag == 3:
			# 	outputs = net(torch.Tensor(inputs).cuda(), torch.Tensor(feature).cuda(), mean2, std2)
			# else:
			# 	outputs = net(torch.Tensor(inputs).cuda())

			# outputs = outputs.cpu().detach().numpy().flatten()[0:len(labels)]
			# loss = criterion(torch.Tensor(outputs).cuda(), torch.Tensor(labels).cuda())
			# valid_loss = loss.item()

			# val_acc_list = []
			# prediction_error = np.abs(outputs-labels)
			# for thresh_hold in thresh_holds:
			# 	acc = np.sum(prediction_error < thresh_hold) / len(prediction_error)
			# 	val_acc_list.append(acc)

			# val_acc = np.mean(val_acc_list)

			# train_acc = 0
			# val_acc = 0

			end = time.time()
			print('[%d/%d][cost time %f] training loss: %.3f, training acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
				(epoch+1, nEpoch, end-start, running_loss / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
			f_log.write("{:d},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, running_loss / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
			f_log.flush()

			save_step = 1
			if BN_flag == 7 or BN_flag == 8 or BN_flag == 10:
				save_step = 1000
			if epoch % save_step == 0:
				torch.save(net.state_dict(), outputPath + 'model_' + str(epoch) + '.pth')

			end = time.time()

			print('--------------------------------- one epoch time: ', end-last_time, ' s, total time: ', end-startall, ' s')
			timing_list.append(end-startall)
			last_time = end

		f_log.close()
		torch.save(net.state_dict(), outputPath + 'model-final_' + str(mid) + '.pth')
		torch.save(net.state_dict(), outputPath + 'model-final.pth')
		print('Finished Training')
		print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! timing_list ', timing_list)
	else:
		modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
		lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
		#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=outputPath+"logs/")

		#plot_model(net, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

		if netType != 5:
			nTrainStep = int(len(yTrainList)/batchSize) + 1
			nValidStep = int(len(yValidList)/batchSize) + 1

			if platform == "win32":
				net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
				verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)
			else:
				net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
				verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)#, \
				# workers=4, use_multiprocessing=True)
		else:
			for [x_batch, y_batch] in trainGenerator:
				print(x_batch.shape)
				print(y_batch.shape)
				print(y_batch[0])
				input_img = tf.convert_to_tensor(x_batch, dtype=tf.float32)
				print(input_img)
				cv2.imshow("input", input_img[0].eval(session=tf.compat.v1.Session())/255)
				imgs = net.g(input_img)

				cv2.imshow("output", imgs[0].eval(session=tf.compat.v1.Session())/255)
				cv2.waitKey(0)

		net.save(outputPath + 'model-final_' + str(mid) + '.h5')
		net.save(outputPath + 'model-final.h5')
	


def train_dnn_multi_two_stream(imageDir_list, labelPath_list, outputPath, netType, flags, specs, modelPath = "", 
	trainRatio = 1.0, partialPreModel = False, reinitHeader = False, 
	BN_flag=0, imageDir_list_advp=[], labelPath_list_advp=[], trainRatio_advp = 1.0, reinitBN = False, 
	pack_flag=False, mid=0, Maxup_flag=False, pytorch_flag=False, nRound=100):
	
	## assigning variables
	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
	batchSize 	   = specs[0]
	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	## prepare the data
	#xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio, specialFilter=True)
	if not pack_flag:
		xList, yList = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio)
		xList_advp, yList_advp = load_train_data_multi(imageDir_list_advp, labelPath_list_advp, nRep, fThreeCameras, trainRatio_advp)
	else:
		xList, yList = load_train_data_multi_pack(imageDir_list, labelPath_list, nRep, fThreeCameras, trainRatio)
		xList_advp, yList_advp = load_train_data_multi_pack(imageDir_list_advp, labelPath_list_advp, nRep, fThreeCameras, trainRatio_advp)

	dict_major2hint = {}
	for x_pair in xList_advp:
		dict_major2hint[x_pair[0]]=x_pair[1]

	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)

	xTrainList_advp, xValidList_advp = train_test_split(np.array(xList_advp), test_size=0.1, random_state=42)
	yTrainList_advp, yValidList_advp = train_test_split(np.array(yList_advp), test_size=0.1, random_state=42)

	## change the data format if necessary
	if fClassifier:
		print('\n######### Classification #########')
		yTrainList = normalize_value(yTrainList)
		yTrainList = to_categorical(yTrainList, num_classes = nClass)
		yValidList = normalize_value(yValidList)
		yValidList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train1 data:', xTrainList.shape, yTrainList.shape)
	print('Valid1 data:', xValidList.shape, yValidList.shape)
	print('Train2 data:', xTrainList_advp.shape, yTrainList_advp.shape)
	print('Valid2 data:', xValidList_advp.shape, yValidList_advp.shape)
	print('##############################\n')
	
	size=(66, 200)
	if (BN_flag == 10): #resnet
		size=(64,64)

	transform=transforms.Compose([
		transforms.Resize(size),
		transforms.ToTensor(),
		# transforms.Lambda(lambda rgb_img: 
		# 	torch.matmul(rgb_img.permute(1, 2, 0), 
		# 	yuv_weight.transpose(0, 1)).permute(2, 0, 1)),
		])

	train_dataset = DrivingDataset_pytorch(xTrainList, yTrainList, transform=transform)
	valid_dataset = DrivingDataset_pytorch(xValidList, yValidList, transform=transform)
	train_dataset_advp = DrivingDataset_pytorch(xTrainList_advp, yTrainList_advp, transform=transform)
	valid_dataset_advp = DrivingDataset_pytorch(xValidList_advp, yValidList_advp, transform=transform)
	#dataset = DrivingDataset_pytorch(xTrainList, yTrainList)
	if platform == "win32":
		# trainGenerator_dst = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		# validGenerator_dst = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		# trainGenerator_src = DataLoader(train_dataset_advp, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		# validGenerator_src = DataLoader(valid_dataset_advp, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		trainGenerator_src = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		validGenerator_src = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		trainGenerator_dst = DataLoader(train_dataset_advp, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
		validGenerator_dst = DataLoader(valid_dataset_advp, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
	else:
		# trainGenerator_dst = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		# validGenerator_dst = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		# trainGenerator_src = DataLoader(train_dataset_advp, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		# validGenerator_src = DataLoader(valid_dataset_advp, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		trainGenerator_src = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		validGenerator_src = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		trainGenerator_dst = DataLoader(train_dataset_advp, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
		validGenerator_dst = DataLoader(valid_dataset_advp, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)


	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	#else:
	#	shutil.rmtree(outputPath)
	#	os.makedirs(outputPath)


	## train
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# net = net.cuda()
	loss_regression = nn.MSELoss()
	loss_domain = nn.NLLLoss()
	#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	#optimizer = optim.Adam(net.parameters(), lr=0.0001)
	train_batch_num = int(len(yTrainList)/batchSize)
	valid_batch_num = int(len(yValidList)/batchSize)
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]


	f_log = open(outputPath + 'loss-log', 'w')
	f_log.write('epoch,loss,loss_t_domain,loss_s_domain,loss_s_label,mean_accuracy,val_loss,val_mean_accuracy\n')

	if BN_flag == 4:
		nChannel = 3
		train_DANN(BN_flag, fClassifier, nClass, nChannel, nEpoch, batchSize, train_batch_num, valid_batch_num, modelPath,
			trainGenerator_src, trainGenerator_dst, validGenerator_src, validGenerator_dst, 
			loss_domain, loss_regression, thresh_holds, outputPath, f_log)
	elif BN_flag == 6:
		nChannel = 3
		train_ADDA(BN_flag, fClassifier, nClass, nChannel, nEpoch, batchSize, train_batch_num, valid_batch_num, modelPath,
			trainGenerator_src, trainGenerator_dst, validGenerator_src, validGenerator_dst, 
			loss_domain, loss_regression, thresh_holds, outputPath, f_log)
	elif BN_flag == 9 or BN_flag == 10:
		nChannel_base = 3
		nChannel_hint = 6
		hard_case_ratio = 1#0.02
		hard_case_ratio_thresh = 1#0.2
		train_hintnet(BN_flag, fClassifier, nClass, nChannel_base, nChannel_hint, nRound, nEpoch, batchSize, train_batch_num, valid_batch_num, modelPath,
			trainGenerator_src, validGenerator_src, imageDir_list, imageDir_list_advp, dict_major2hint, transform, loss_regression, thresh_holds, outputPath, f_log, hard_case_ratio, hard_case_ratio_thresh)

	f_log.close()

def train_hintnet(BN_flag, fClassifier, nClass, nChannel_base, nChannel_hint, nRound, nEpoch, batchSize, train_batch_num, valid_batch_num, modelPath,
	trainGenerator_base, validGenerator_base, imageDir_list_base, imageDir_list_hint, dict_major2hint, transform, criterion, thresh_holds, outputPath, f_log, hard_case_ratio = 0.01, hard_case_ratio_thresh=0.1):

	net_f_ori, net_f_hint, net_d = create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel=nChannel_base, nChannel_hint=nChannel_hint)

	net_f_ori.cuda()
	net_f_hint.cuda()
	net_d.cuda()

	optimizer_fo = optim.Adam(net_f_ori.parameters(), lr=0.0001)
	optimizer_fh = optim.Adam(net_f_hint.parameters(), lr=0.0001)
	optimizer_d = optim.Adam(net_d.parameters(), lr=0.0001)

	dict_hard_cover = {}

	for round_id in range(nRound):

		img_path_list = []
		label_list = []
		error_list = []
		# train all cases for nEpoch epochs
		for epoch in range(nEpoch):
			start = time.time()

			data_base_iter = iter(trainGenerator_base)

			net_f_ori.train()
			net_d.train()

			running_loss = 0
			train_acc_list = [0,0,0,0,0,0]
			sample_count = 0

			for i in range(len(trainGenerator_base)):
				# training model using source data
				img_base, label_base, img_path = data_base_iter.next()
				# img_path = "a"
				# s_label = s_label / 15

				# batch_size = len(img_base)

				feature_base = net_f_ori(input_data=img_base.cuda(non_blocking=True))
				output_base,_ = net_d(feature_base)

				labels_2d = label_base.reshape(output_base.shape)

				optimizer_fo.zero_grad()
				optimizer_d.zero_grad()

				loss = criterion(output_base, torch.Tensor(labels_2d).cuda(non_blocking=True))

				loss.backward()

				optimizer_fo.step()
				optimizer_d.step()

				# print statistics
				running_loss += loss.item()

				prediction_error = np.abs(output_base.cpu().detach().numpy().flatten()-label_base.numpy().flatten())
				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error < thresh_hold)
					train_acc_list[j] += acc_count

				sample_count += len(prediction_error)

				if epoch == nEpoch-1:
					# save the result in the last epoch
					img_path_list = img_path_list + img_path
					error_list = error_list + prediction_error.tolist()
					label_list = label_list + label_base.detach().numpy().tolist()

				# if i >= train_batch_num-1:
				# 	break

			for j in range(len(train_acc_list)):
				train_acc_list[j] = train_acc_list[j] / sample_count
			train_acc = np.mean(train_acc_list)

			valid_loss = 0.0
			valid_acc_list = [0,0,0,0,0,0]
			net_f_ori.eval()
			net_d.eval()

			valid_iter = iter(validGenerator_base)

			# for i, (inputs, labels, _) in enumerate(validGenerator_src):
			# 	feature = net_s(input_data=inputs.cuda(non_blocking=True))
			# for i, (inputs, labels, _) in enumerate(validGenerator_base):
			for i in range(len(validGenerator_base)):
				inputs, labels, _ = valid_iter.next()
				feature = net_f_ori(input_data=inputs.cuda(non_blocking=True))
				outputs,_ = net_d(feature)

				labels = labels.numpy().flatten()
				# labels = labels / 15
				labels_2d = labels.reshape((labels.shape[0], 1))

				loss = criterion(outputs, torch.Tensor(labels_2d).cuda())

				# print statistics
				valid_loss += loss.item()

				prediction_error = np.abs(outputs.cpu().detach().numpy().flatten()-labels)
				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error < thresh_hold)
					valid_acc_list[j] += acc_count

				# if i >= valid_batch_num-1:
				# 	break

			for j in range(len(valid_acc_list)):
				valid_acc_list[j] = valid_acc_list[j] / valid_batch_num / batchSize
			val_acc = np.mean(valid_acc_list)

			end = time.time()
			print('[%d/%d %d/%d][cost time %f] train loss: %.3f, train acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
				(round_id+1, nRound, epoch+1, nEpoch, end-start, running_loss/train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
			# print('[%d/%d][cost time %f] training loss: %.3f, training acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
			# 	(epoch+1, nEpoch, end-start, running_loss / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
			f_log.write("{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(round_id+1, epoch+1, running_loss / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
			f_log.flush()

		# 10% round general training first, then begin hintnet training
		# if round_id < nRound * 0.1:
		# 	continue

		if round_id == nRound-1:
			# no need to learn hard cases in the last round, avoid bias to hard cases
			break

		# figure out the hard cases
		sorted_index_list = np.argsort(error_list)
		total_number = len(error_list)
		hard_case_number = int(total_number * hard_case_ratio)

		for i in range(hard_case_number):
			if len(dict_hard_cover) >= hard_case_ratio_thresh * total_number:
				break

			# idx = sorted_index_list[total_number-1-i] # hard cases

			idx = int(total_number / hard_case_number * i) # uniform distribution
			idx = np.clip(idx, 0, total_number-1)
			idx = sorted_index_list[idx]

			img_path = img_path_list[idx]
			dict_hard_cover[img_path] = label_list[idx][0]

		hard_case_list = []
		hard_case_label_list = []
		for img_path in dict_hard_cover:
			label = dict_hard_cover[img_path]
			element1 = [img_path, img_path, dict_major2hint[img_path]]

			# for new_folder in imageDir_list_hint:
			# 	element1.append(img_path.replace(imageDir_list_base[0], new_folder))
			# element1[-1] = element1[-1].replace("_camera_", "_label_") # for segall

			hard_case_list.append(element1)
			hard_case_label_list.append(label)

		hard_case_dataset = DrivingDataset_pytorch(hard_case_list, hard_case_label_list, transform=transform)
		# batch_size = 64

		if platform == "win32":
			hard_case_generator = DataLoader(hard_case_dataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)
			# hard_case_generator = trainGenerator_base
		else:
			hard_case_generator = DataLoader(hard_case_dataset, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True)
			# hard_case_generator = trainGenerator_base

		# copy net_f_ori parameter to net_f_hint
		if BN_flag == 9:
			net_f_hint.conv1.weight.data = torch.zeros(net_f_hint.conv1.weight.data.shape)
			net_f_hint.conv1.weight.data[:,0:3,:,:] = net_f_ori.conv1.weight.data
			net_f_hint.conv1.bias.data = net_f_ori.conv1.bias.data
			net_f_hint.conv2.weight.data = net_f_ori.conv2.weight.data
			net_f_hint.conv2.bias.data = net_f_ori.conv2.bias.data
			net_f_hint.conv3.weight.data = net_f_ori.conv3.weight.data
			net_f_hint.conv3.bias.data = net_f_ori.conv3.bias.data
			net_f_hint.conv4.weight.data = net_f_ori.conv4.weight.data
			net_f_hint.conv4.bias.data = net_f_ori.conv4.bias.data
			net_f_hint.conv5.weight.data = net_f_ori.conv5.weight.data
			net_f_hint.conv5.bias.data = net_f_ori.conv5.bias.data
		elif BN_flag == 10:
			net_f_hint.conv1.weight.data = torch.zeros(net_f_hint.conv1.weight.data.shape)
			net_f_hint.conv1.weight.data[:,0:3,:,:] = net_f_ori.conv1.weight.data
			net_f_hint.conv1.bias.data = net_f_ori.conv1.bias.data
			net_f_hint.resnet152.load_state_dict(net_f_ori.resnet152.state_dict())

		net_f_hint.cuda()

		net_f_ori.train()
		net_f_hint.train()
		net_d.train()

		print('[%d/%d] hard case number: %d' % (round_id+1, nRound, len(dict_hard_cover)))
		# print('[%d/%d][cost time %f] training loss: %.3f, training acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
		# 	(epoch+1, nEpoch, end-start, running_loss / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
		f_log.write("{:d},{:d}\n".format(round_id+1, len(dict_hard_cover)))
		f_log.flush()

		# train hard cases for nEpoch epochs
		for epoch in range(nEpoch):
			start = time.time()

			running_loss_hard = 0
			loss1 = 0
			loss2 = 0
			loss3 = 0
			train_acc_list = [0,0,0,0,0,0]
			train_acc_list_hint = [0,0,0,0,0,0]
			sample_count = 0
			batch_num_hard = len(hard_case_generator)

			data_hint_iter = iter(hard_case_generator)

			for i in range(len(hard_case_generator)):
				# training model using source data
				img_all, labels, _ = data_hint_iter.next()
				# print(img_all)
				# print(labels)

				img_base = img_all[:,0:3,:,:]
				img_hint = img_all[:,3:9,:,:]
				# s_label = s_label / 15

				feature_base = net_f_ori(input_data=img_base.cuda(non_blocking=True))
				output_base,_ = net_d(feature_base)

				feature_hint = net_f_hint(input_data=img_hint.cuda(non_blocking=True))
				output_hint,_ = net_d(feature_hint)

				labels_2d = labels.reshape(output_base.shape)

				optimizer_fo.zero_grad()
				optimizer_fh.zero_grad()
				optimizer_d.zero_grad()

				# ************************ teacher and student learn together ************************
				# loss_1 = criterion(output_base, torch.Tensor(labels_2d).cuda(non_blocking=True))
				# loss_2 = criterion(output_hint, torch.Tensor(labels_2d).cuda(non_blocking=True))
				# loss_3 = criterion(feature_base, feature_hint)

				# loss = loss_1 + loss_2 + loss_3

				# loss.backward()

				# optimizer_fo.step()
				# optimizer_fh.step()
				# optimizer_d.step()



				# ************************ teacher first, then student ************************
				# teacher learn first
				loss_2 = criterion(output_hint, torch.Tensor(labels_2d).cuda(non_blocking=True))
				loss_2.backward(retain_graph=True)
				optimizer_fh.step()

				# student learn next
				loss_1 = criterion(output_base, torch.Tensor(labels_2d).cuda(non_blocking=True))
				loss_3 = criterion(feature_base, feature_hint)

				loss = loss_1 + loss_3
				loss.backward()

				optimizer_fo.step()
				optimizer_d.step()



				# print statistics
				running_loss_hard += loss.item()
				loss1 += loss_1.item()
				loss2 += loss_2.item()
				loss3 += loss_3.item()

				prediction_error = np.abs(output_base.cpu().detach().numpy().flatten()-labels.numpy().flatten())
				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error < thresh_hold)
					train_acc_list[j] += acc_count

				prediction_error_hint = np.abs(output_hint.cpu().detach().numpy().flatten()-labels.numpy().flatten())
				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error_hint < thresh_hold)
					train_acc_list_hint[j] += acc_count

				sample_count += len(prediction_error)

			for j in range(len(train_acc_list)):
				train_acc_list[j] = train_acc_list[j] / sample_count
			train_acc = np.mean(train_acc_list)

			for j in range(len(train_acc_list_hint)):
				train_acc_list_hint[j] = train_acc_list_hint[j] / sample_count
			train_acc_hint = np.mean(train_acc_list_hint)

			valid_loss = 0.0
			valid_acc_list = [0,0,0,0,0,0]
			net_f_ori.eval()
			net_d.eval()

			valid_iter = iter(validGenerator_base)

			# for i, (inputs, labels, _) in enumerate(validGenerator_src):
			# 	feature = net_s(input_data=inputs.cuda(non_blocking=True))
			# for i, (inputs, labels, _) in enumerate(validGenerator_base):
			for i in range(len(validGenerator_base)):
				inputs, labels, _ = valid_iter.next()
				feature = net_f_ori(input_data=inputs.cuda(non_blocking=True))
				outputs,_ = net_d(feature)

				labels = labels.numpy().flatten()
				# labels = labels / 15
				labels_2d = labels.reshape((labels.shape[0], 1))

				loss = criterion(outputs, torch.Tensor(labels_2d).cuda())

				# print statistics
				valid_loss += loss.item()

				prediction_error = np.abs(outputs.cpu().detach().numpy().flatten()-labels)
				for j,thresh_hold in enumerate(thresh_holds):
					acc_count = np.sum(prediction_error < thresh_hold)
					valid_acc_list[j] += acc_count

				# if i >= valid_batch_num-1:
				# 	break

			for j in range(len(valid_acc_list)):
				valid_acc_list[j] = valid_acc_list[j] / valid_batch_num / batchSize
			val_acc = np.mean(valid_acc_list)

			end = time.time()
			print('[%d/%d %d/%d][cost time %f] loss overall: %.3f, loss base: %.3f, loss hint: %.3f, loss feature: %.3f, training acc base: %.3f, training acc hint: %.3f, valid loss: %.3f, valid acc: %.3f' % \
				(round_id+1, nRound, epoch+1, nEpoch, end-start, running_loss_hard/batch_num_hard, loss1/batch_num_hard, loss2/batch_num_hard, loss3/batch_num_hard, train_acc, train_acc_hint, valid_loss / valid_batch_num, val_acc))
			# print('[%d/%d][cost time %f] training loss: %.3f, training acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
			# 	(epoch+1, nEpoch, end-start, running_loss / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
			f_log.write("{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(round_id+1, epoch+1, running_loss_hard/batch_num_hard, loss1/batch_num_hard, loss2/batch_num_hard, loss3/batch_num_hard, train_acc, train_acc_hint, valid_loss / valid_batch_num, val_acc))
			f_log.flush()

		save_step = 1
		if nEpoch == 1 or BN_flag == 10:
			save_step = 100
		if round_id % save_step == 0:
			torch.save(net_f_ori.state_dict(), outputPath + 'model_f_ori_' + str(round_id) + '.pth')
			torch.save(net_f_hint.state_dict(), outputPath + 'model_f_hint_' + str(round_id) + '.pth')
			torch.save(net_d.state_dict(), outputPath + 'model_d_' + str(round_id) + '.pth')

	torch.save(net_f_ori.state_dict(), outputPath + 'model_f_ori-final.pth')
	torch.save(net_f_hint.state_dict(), outputPath + 'model_f_hint-final.pth')
	torch.save(net_d.state_dict(), outputPath + 'model_d-final.pth')
	print('Finished Training')


def load_pretrain_model(fClassifier, nClass, nChannel, modelPath):
	net = create_nvidia_network_pytorch(0, fClassifier, nClass, nChannel)
	print(net)

	print("pretrain modelPath: ", modelPath)
	net.load_state_dict(torch.load(modelPath))


	net_s = ADDA_NVIDIA_FEATURE_CNN(nChannel)
	net_t = ADDA_NVIDIA_FEATURE_CNN(nChannel)
	net_r = ADDA_NVIDIA_REGRESSOR()

	net_s.conv1 = net.conv1
	net_s.conv2 = net.conv2
	net_s.conv3 = net.conv3
	net_s.conv4 = net.conv4
	net_s.conv5 = net.conv5
	net_s.fc1 = net.fc1

	net_t.conv1 = net.conv1
	net_t.conv2 = net.conv2
	net_t.conv3 = net.conv3
	net_t.conv4 = net.conv4
	net_t.conv5 = net.conv5
	net_t.fc1 = net.fc1

	net_r.fc1 = net.fc2
	net_r.fc2 = net.fc3
	net_r.fc3 = net.fc4

	return net_s, net_t, net_r, net


def train_ADDA(BN_flag, fClassifier, nClass, nChannel, nEpoch, batchSize, train_batch_num, valid_batch_num, modelPath,
	trainGenerator_src, trainGenerator_dst, validGenerator_src, validGenerator_dst, 
	loss_domain, loss_regression, thresh_holds, outputPath, f_log):

	net_s, net_t, net_r, net_pretrained = load_pretrain_model(fClassifier, nClass, nChannel, modelPath)

	net_t.cuda()
	net_s.cuda()
	net_r.cuda()

	net_s.eval()
	net_r.eval()

	for param in net_s.parameters():
		param.requires_grad = False
	for param in net_r.parameters():
		param.requires_grad = False

	# net_t = ADDA_NVIDIA_FEATURE_CNN(nChannel)
	net_d = ADDA_DOMAIN_DISCRIMINATOR()
	net_d.cuda()

	optimizer_t = optim.Adam(net_t.parameters(), lr=0.001)
	optimizer_d = optim.Adam(net_d.parameters(), lr=0.0001)

	for epoch in range(nEpoch):
		start = time.time()

		len_dataloader = min(len(trainGenerator_src), len(trainGenerator_dst)) - 1
		data_source_iter = iter(trainGenerator_src)
		data_target_iter = iter(trainGenerator_dst)

		net_t.train()
		net_d.train()

		i = 0
		running_loss = 0
		running_loss_adv = 0
		train_acc_list = [0,0,0,0,0,0]
		loss_t_domain = 0
		loss_s_domain = 0
		loss_s_label = 0
		while i < len_dataloader:

			net_d.zero_grad()
			net_t.zero_grad()

			# training model using source data
			data_source = data_source_iter.next()
			s_img, s_label = data_source
			# s_label = s_label / 15

			batch_size = len(s_img)

			regression_label = torch.FloatTensor(batch_size)
			s_domain_label = torch.zeros(batch_size)
			s_domain_label = s_domain_label.long()

			s_img = s_img.cuda()
			s_label = s_label.cuda()
			regression_label = regression_label.cuda()
			s_domain_label = s_domain_label.cuda()

			regression_label.resize_as_(s_label).copy_(s_label)

			s_feature = net_s(input_data=s_img.cuda(non_blocking=True))
			s_domain = net_d(s_feature)
			s_regression = net_r(s_feature)

			err_s_domain = loss_domain(s_domain, s_domain_label)

			# training model using target data
			data_target = data_target_iter.next()
			t_img, t_label = data_target
			# t_label = t_label / 15

			batch_size = len(t_img)

			regression_label = torch.FloatTensor(batch_size)
			t_domain_label = torch.ones(batch_size)
			t_domain_label = t_domain_label.long()

			t_img = t_img.cuda()
			t_label = t_label.cuda()
			regression_label = regression_label.cuda()
			t_domain_label = t_domain_label.cuda()

			regression_label.resize_as_(t_label).copy_(t_label)

			t_feature = net_t(input_data=t_img.cuda(non_blocking=True))
			t_domain = net_d(t_feature)
			t_regression = net_r(t_feature)

			err_t_domain = loss_domain(t_domain, t_domain_label)

			err = err_t_domain + err_s_domain
			err.backward()

			optimizer_d.step()


			# *************** optimize feature network to confuse the net_d *******************
			net_d.zero_grad()
			net_t.zero_grad()

			t_feature = net_t(input_data=t_img.cuda(non_blocking=True))
			t_domain = net_d(t_feature)
			err_adv = loss_domain(t_domain, s_domain_label)
			err_adv.backward()

			optimizer_t.step()
			# *************** optimize feature network *******************

			# print statistics
			running_loss += err.item()
			running_loss_adv += err_adv.item()
			loss_t_domain += err_t_domain.item()
			loss_s_domain += err_s_domain.item()
			loss_s_label += 0

			# pre_regression = net_pretrained(s_img.cuda(non_blocking=True))
			# prediction_error = np.abs(s_regression.cpu().detach().numpy().flatten()-s_label.cpu().detach().numpy().flatten())
			prediction_error = np.abs(t_regression.cpu().detach().numpy().flatten()-t_label.cpu().detach().numpy().flatten())
			for j,thresh_hold in enumerate(thresh_holds):
				acc_count = np.sum(prediction_error < thresh_hold)
				train_acc_list[j] += acc_count

			i += 1

			# print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
			# 	% (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
			# 	err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

		for j in range(len(train_acc_list)):
			train_acc_list[j] = train_acc_list[j] / len_dataloader / batchSize
		train_acc = np.mean(train_acc_list)


		valid_loss = 0.0
		valid_acc_list = [0,0,0,0,0,0]
		net_d.eval()
		net_r.eval()

		# for i, (inputs, labels, _) in enumerate(validGenerator_src):
		# 	feature = net_s(input_data=inputs.cuda(non_blocking=True))
		for i, (inputs, labels, _) in enumerate(validGenerator_dst):
			feature = net_t(input_data=inputs.cuda(non_blocking=True))
			domain = net_d(feature)
			outputs = net_r(feature)

			labels = labels.numpy().flatten()
			# labels = labels / 15
			labels_2d = labels.reshape((labels.shape[0], 1))

			loss = loss_regression(outputs, torch.Tensor(labels_2d).cuda())

			# print statistics
			valid_loss += loss.item()

			prediction_error = np.abs(outputs.cpu().detach().numpy().flatten()-labels)
			for j,thresh_hold in enumerate(thresh_holds):
				acc_count = np.sum(prediction_error < thresh_hold)
				valid_acc_list[j] += acc_count

			if i >= valid_batch_num-1:
				break

		for j in range(len(valid_acc_list)):
			valid_acc_list[j] = valid_acc_list[j] / valid_batch_num / batchSize
		val_acc = np.mean(valid_acc_list)

		end = time.time()
		print('[%d/%d][cost time %f] training loss: %.3f, loss_t_domain: %.3f, loss_s_domain: %.3f, loss_s_label: %.3f, loss_t_adv: %.3f, training acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
			(epoch+1, nEpoch, end-start, running_loss / train_batch_num, loss_t_domain, loss_s_domain, loss_s_label, running_loss_adv / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
		f_log.write("{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, running_loss / train_batch_num, loss_t_domain, loss_s_domain, loss_s_label, running_loss_adv / train_batch_num, train_acc, valid_loss / valid_batch_num, val_acc))
		f_log.flush()
		if epoch % 100 == 0:
			torch.save(net_d.state_dict(), outputPath + 'model_d_' + str(epoch) + '.pth')
			torch.save(net_r.state_dict(), outputPath + 'model_r_' + str(epoch) + '.pth')

	torch.save(net_d.state_dict(), outputPath + 'model_d-final.pth')
	torch.save(net_r.state_dict(), outputPath + 'model_r-final.pth')
	print('Finished Training')


def train_DANN(BN_flag, fClassifier, nClass, nChannel, nEpoch, batchSize, train_batch_num, valid_batch_num, modelPath,
	trainGenerator_src, trainGenerator_dst, validGenerator_src, validGenerator_dst, 
	loss_domain, loss_regression, thresh_holds, outputPath, f_log):

	#net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
	net = create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel)

	optimizer = optim.Adam(net.parameters(), lr=0.0001)

	for epoch in range(nEpoch):
		start = time.time()

		len_dataloader = min(len(trainGenerator_src), len(trainGenerator_dst)) - 1
		data_source_iter = iter(trainGenerator_src)
		data_target_iter = iter(trainGenerator_dst)

		net.train()

		i = 0
		running_loss = 0
		train_acc_list = [0,0,0,0,0,0]
		loss_t_domain = 0
		loss_s_domain = 0
		loss_s_label = 0
		while i < len_dataloader:

			p = float(i + epoch * len_dataloader) / nEpoch / len_dataloader
			alpha = 2. / (1. + np.exp(-10 * p)) - 1

			net.zero_grad()

			# training model using source data
			data_source = data_source_iter.next()
			s_img, s_label = data_source
			s_img = np.transpose(s_img, (0, 3, 1, 2))

			batch_size = len(s_img)

			input_img = torch.FloatTensor(batch_size, 3, 266, 400)
			regression_label = torch.FloatTensor(batch_size)
			domain_label = torch.zeros(batch_size)
			domain_label = domain_label.long()

			s_img = s_img.cuda()
			s_label = s_label.cuda()
			input_img = input_img.cuda()
			regression_label = regression_label.cuda()
			domain_label = domain_label.cuda()

			input_img.resize_as_(s_img).copy_(s_img)
			regression_label.resize_as_(s_label).copy_(s_label)

			regression_output, domain_output = net(input_data=input_img, alpha=alpha)
			err_s_label = loss_regression(regression_output, regression_label)
			err_s_domain = loss_domain(domain_output, domain_label)

			# training model using target data
			data_target = data_target_iter.next()
			t_img, t_label = data_target
			t_img = np.transpose(t_img, (0, 3, 1, 2))

			batch_size = len(t_img)

			input_img = torch.FloatTensor(batch_size, 3, 266, 400)
			regression_label = torch.FloatTensor(batch_size)
			domain_label = torch.ones(batch_size)
			domain_label = domain_label.long()

			t_img = t_img.cuda()
			t_label = t_label.cuda()
			input_img = input_img.cuda()
			regression_label = regression_label.cuda()
			domain_label = domain_label.cuda()

			input_img.resize_as_(t_img).copy_(t_img)
			regression_label.resize_as_(t_label).copy_(t_label)

			t_regression_output, t_domain_output = net(input_data=input_img, alpha=alpha)
			err_t_label = loss_regression(t_regression_output, regression_label)
			err_t_domain = loss_domain(domain_output, domain_label)


			# err = err_t_domain + err_t_label + err_s_domain + err_s_label
			err = err_t_domain + err_s_domain + err_s_label
			loss_t_domain += err_t_domain.item()
			loss_s_domain += err_s_domain.item()
			loss_s_label += err_s_label.item()
			err.backward()
			optimizer.step()

			# print statistics
			running_loss += err.item()

			prediction_error = np.abs(t_regression_output.cpu().detach().numpy().flatten()-t_label.cpu().detach().numpy().flatten())
			for j,thresh_hold in enumerate(thresh_holds):
				acc_count = np.sum(prediction_error < thresh_hold)
				train_acc_list[j] += acc_count

			i += 1

			# print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
			# 	% (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
			# 	err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

		for j in range(len(train_acc_list)):
			train_acc_list[j] = train_acc_list[j] / len_dataloader / batchSize
		train_acc = np.mean(train_acc_list)


		valid_loss = 0.0
		valid_acc_list = [0,0,0,0,0,0]
		net.eval()

		for i, (inputs, labels, _) in enumerate(validGenerator_dst):
			labels = labels.numpy().flatten()
			if BN_flag == 4:
				inputs = np.transpose(inputs, (0, 3, 1, 2))
				regression_output, domain_output = net(input_data=torch.Tensor(inputs).cuda(), alpha=alpha)
				outputs = regression_output
			elif BN_flag == 3:
				image, feature = inputs
				labels1, labels2 = labels
				image = np.transpose(image, (0, 3, 1, 2))
				inputs = [image, feature]
				labels = np.concatenate((labels1, labels2))
				outputs = net(torch.Tensor(image).cuda(), torch.Tensor(feature).cuda(), mean2, std2)
			else:
				inputs = np.transpose(inputs, (0, 3, 1, 2))
				outputs = net(torch.Tensor(inputs).cuda())

			labels = labels.flatten()
			labels_2d = labels.reshape((labels.shape[0], 1))

			loss = loss_regression(outputs, torch.Tensor(labels_2d).cuda())

			# print statistics
			valid_loss += loss.item()

			prediction_error = np.abs(outputs.cpu().detach().numpy().flatten()-labels)
			for j,thresh_hold in enumerate(thresh_holds):
				acc_count = np.sum(prediction_error < thresh_hold)
				valid_acc_list[j] += acc_count

			if i >= valid_batch_num-1:
				break

		for j in range(len(valid_acc_list)):
			valid_acc_list[j] = valid_acc_list[j] / valid_batch_num / batchSize
		val_acc = np.mean(valid_acc_list)


		# (inputs, labels, _) = load_data_all(xValidList, yValidList)
		# inputs = np.transpose(inputs, (0, 3, 1, 2))
		# labels_2d = labels.reshape((labels.shape[0], 1))

		# if BN_flag == 4:
		# 	regression_output, domain_output = net(input_data=torch.Tensor(inputs).cuda(), alpha=1)
		# 	outputs = regression_output
		# elif BN_flag == 3:
		# 	outputs = net(torch.Tensor(inputs).cuda(), torch.Tensor(feature).cuda(), mean2, std2)
		# else:
		# 	outputs = net(torch.Tensor(inputs).cuda())

		# outputs = outputs.cpu().detach().numpy().flatten()[0:len(labels)]
		# loss = loss_regression(torch.Tensor(outputs).cuda(), torch.Tensor(labels).cuda())
		# valid_loss = loss.item()

		# val_acc_list = []
		# prediction_error = np.abs(outputs-labels)
		# for thresh_hold in thresh_holds:
		# 	acc = np.sum(prediction_error < thresh_hold) / len(prediction_error)
		# 	val_acc_list.append(acc)

		# val_acc = np.mean(val_acc_list)


		end = time.time()
		print('[%d/%d][cost time %f] training loss: %.3f, loss_t_domain: %.3f, loss_s_domain: %.3f, loss_s_label: %.3f, training acc: %.3f, valid loss: %.3f, valid acc: %.3f' % \
			(epoch+1, nEpoch, end-start, running_loss / train_batch_num, loss_t_domain, loss_s_domain, loss_s_label, train_acc, valid_loss / valid_batch_num, val_acc))
		f_log.write("{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch+1, running_loss / train_batch_num, loss_t_domain, loss_s_domain, loss_s_label, train_acc, valid_loss / valid_batch_num, val_acc))
		f_log.flush()
		if epoch % 100 == 0:
			torch.save(net.state_dict(), outputPath + 'model_' + str(epoch) + '.pth')

	torch.save(net.state_dict(), outputPath + 'model-final.pth')
	print('Finished Training')



	
'''
def train_nv_icra19(trainPath, trainSet, repSet, outputPath, batchSize, nEpoch):
	
	## prepare the data
	xList = [];
	yList = [];
	for i in range(len(repSet)):
		print(trainPath + trainSet[i] + '/')
		xTmp, yTmp = load_train_data(trainPath + trainSet[i] + '/', repSet[i], False)
		xList = xList + xTmp;
		yList = yList + yTmp;
		
	xTrainList, xValidList = train_test_split(np.array(xList), test_size=0.1, random_state=42)
	yTrainList, yValidList = train_test_split(np.array(yList), test_size=0.1, random_state=42)
	
	print('\n######### Regression #########')
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('##############################\n')
		
	outputPath = outputPath + 'output/models-cnn/';
	net = net_nvidia(False, -1)
	trainGenerator = gen_train_data_random(xTrainList, yTrainList, batchSize)
	validGenerator = gen_train_data_random(xValidList, yValidList, batchSize)
	
	## setup outputs
	if not os.path.exists(outputPath):
		os.makedirs(outputPath)
	else:
		shutil.rmtree(outputPath)
		os.makedirs(outputPath)
	modelLog = ModelCheckpoint(outputPath + 'model{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
	lossLog  = CSVLogger(outputPath + 'loss-log', append=True, separator=',')
	
	## train
	nTrainStep = int(len(yTrainList)/batchSize)
	nValidStep = int(len(yValidList)/batchSize)
	net.fit_generator(trainGenerator, steps_per_epoch=nTrainStep, epochs=nEpoch, \
	verbose=2, callbacks=[modelLog,lossLog], validation_data=validGenerator, validation_steps=nValidStep)
	#net.save(outputFolder + 'model-final.h5')
	print(net.summary())
'''
	
	
	
'''
def train_dnn_overfitting(trainSpec, xTrainList, yTrainList, xValidList, yValidList):	
	## assign variables
	outputFolder = trainSpec[0]
	batchSize	= trainSpec[1]
	nEpoch	   = trainSpec[2]
	isClassify   = trainSpec[3]
	nClass	   = trainSpec[4]
	randomDistortFlag = False
	
	## pulling out 128 random samples and training just on them, to make sure the model is capable of overfitting
	tmpIndices = np.random.randint(0, len(xTrainList), 128)
	xTrainList = xTrainList[tmpIndices]
	yTrainList = yTrainList[tmpIndices]
	
	tmpIndices = np.random.randint(0, len(xValidList), 12)
	xValidList = xValidList[tmpIndices]
	yValidList = yValidList[tmpIndices]
	
	
	X,y = ([],[])  
	for i in range(len(yTrainList)):
		img = resize_image(cv2.imread(xTrainList[i]))
		angle = yTrainList[i]
		X.append(img)
		y.append(angle)
	X = np.array(X)
	y = np.array(y)
	
	X_val, y_val = ([],[]) 
	for i in range(len(yValidList)):
		img = resize_image(cv2.imread(xValidList[i]))
		angle = yValidList[i]
		X_val.append(img)
		y_val.append(angle)
	X_val = np.array(X_val)
	y_val = np.array(y_val)
		
		
	## get a network
	net = net_testing(isClassify, nClass)

	## initialize generators
	if isClassify:
		print('\n######### Classification #########')
		trainLabelList = to_categorical(yTrainList, num_classes = nClass)
		validLabelList = to_categorical(yValidList, num_classes = nClass)
	else:
		print('\n######### Regression #########')
		
	print('Train data:', xTrainList.shape, yTrainList.shape)
	print('Valid data:', xValidList.shape, yValidList.shape)
	print('############################\n')
		

	# Fit the model
	history = net.fit(X, y, epochs=10, verbose=2, validation_data = (X_val,y_val), shuffle = True)
	net.save(outputFolder + 'model-final.h5')
	#print(net.summary())

'''

def get_bin_id(v, bin_size=1, bin_amount=49):
	return np.clip(int(v / bin_size + bin_amount / 2), 0, 48)

def generate_bin_border_list():
	step = 0.2
	start_b = 0.1
	current_b = start_b
	bin_border_list_1 = [current_b]
	while current_b < 25:
		step *= 2
		current_b = current_b + step
		bin_border_list_1.append(current_b)
		#print(current_b)

	bin_border_list = []
	m = len(bin_border_list_1)
	for i in range(m-1):
		bin_border_list.append(-bin_border_list_1[m-2-i])
	for i in range(m-1):
		bin_border_list.append(bin_border_list_1[i])

	# for i in range(len(bin_border_list)):
	# 	print(bin_border_list[i])

	return bin_border_list

def get_bin_id_ununiform(v, bin_border_list):

	for i in range(len(bin_border_list)):
		if v < bin_border_list[i]:
			break

	return i

def bin_metric(predict_results, groundtruth_labels, uniform_bin=False, thresh_hold=0.2):
	if uniform_bin:
		bin_size = 1
		bin_amount = 49
	else:
		bin_amount = 13
	total_array = np.zeros(bin_amount)
	correct_array = np.zeros(bin_amount)

	bin_border_list = generate_bin_border_list()

	for i in range(len(groundtruth_labels)):
		v_gt = groundtruth_labels[i]
		if uniform_bin:
			bin_id = get_bin_id(v_gt, bin_size, bin_amount)
		else:
			bin_id = get_bin_id_ununiform(v_gt, bin_border_list)
		total_array[bin_id] += 1

		v_p = predict_results[i]
		if (abs(v_p-v_gt) < thresh_hold):
			correct_array[bin_id] += 1

	for i in range(bin_amount):
		if int(total_array[i]) == 0:
			print(i, ': ', int(correct_array[i]), ' / ', int(total_array[i]), '  0%')
		else:
			print(i, ': ', int(correct_array[i]), ' / ', int(total_array[i]), '  {:.2f}'.format(correct_array[i]/total_array[i]*100), '%')
		# if i == int(bin_amount/2):
		# 	print('(', int(correct_array[i]), '/', int(total_array[i]), ')', end = ' ')
		# else:
		# 	print(int(correct_array[i]), '/', int(total_array[i]), end = ' ')



	
def test_dnn_multi(modelPath, imageDir_list, labelPath_list, outputPath, netType, flags, specs, BN_flag=0, pathID=0, ratio=1, pack_flag=False, pytorch_flag=False, net=""):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]

	specialFilter = 0
	if BN_flag == 7:
		specialFilter = 2
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	if not pack_flag:
		testImagePaths, testLabels = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, ratio=ratio, specialFilter=specialFilter)
	else:
		testImagePaths, testLabels = load_train_data_multi_pack(imageDir_list, labelPath_list, nRep, fThreeCameras, ratio=ratio)

	#testImagePaths, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras, ratio=ratio)
	testImagePaths = np.array(testImagePaths)
	testLabels = np.array(testLabels)
	if pack_flag:
		testLabels = testLabels[:,0]
	n = len(testLabels)

	if fClassifier:
		print('\n######### Classification #########')
		testLabels = normalize_value(testLabels)
		testLabels = to_categorical(testLabels, num_classes = nClass)
	
	print(testImagePaths)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')

	testData = []
	if not pack_flag:
		for i in range(len(testLabels)):
			image_path = testImagePaths[i]
			if not os.path.isfile(image_path):
				image_path = image_path.replace(".jpg", "_fake.png")
			img = resize_image(cv2.imread(image_path))
			testData.append(img)
	else:
		for i in range(len(testImagePaths)):
			for j in range(len(testImagePaths[i])):
				image_path = testImagePaths[i][j]
				if not os.path.isfile(image_path):
					image_path = image_path.replace(".jpg", "_fake.png")
				img_1 = resize_image(cv2.imread(image_path))
				if j == 0:
					img = img_1
				else:
					img = np.concatenate((img, img_1), axis=2)

			#noise = np.random.uniform(low=0, high=255, size=(img.shape[0], img.shape[1], 1))
			#img = np.concatenate((img, noise), axis=2)
			testData.append(img)

	testData = np.array(testData)
	nChannel=3
	if pack_flag:
		testImagePaths = testImagePaths[:,0]
		nChannel = 3*len(imageDir_list)

	# redistribution
	#testData = ((((testData / 127.5 - 1) - (-0.030626608) ) / 0.3224381 * 0.320617 + (-0.07968183)) + 1) * 127.5
	# print(np.min(testData))
	# print(np.max(testData))

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	empty_net_flag = False
	if net == "":
		empty_net_flag = True
		if pytorch_flag:
			net = create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel)
		else:
			if netType == 1:
		# 		outputPath = trainPath + 'trainedModels/models-cnn/';
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
				if BN_flag == 3:
					net = net[0]
			elif netType == 2:
		# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
				net = net_lstm(2, nFramesSample)
			elif netType == 3:
		# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
				net = net_lstm(3, nFramesSample)

		## load model weights
		if modelPath != "":
			if pytorch_flag:
				net.load_state_dict(torch.load(modelPath))
				net.eval()
				net.cuda()
			else:
				net.load_weights(modelPath)

	#print(net.layers[3].get_weights())
	if pytorch_flag:
		print(net)
	else:
		print(net.summary())

	# inference
	if pytorch_flag:
		testData = np.transpose(testData, (0, 3, 1, 2))
		predictResults = net(torch.Tensor(testData).cuda())
		if BN_flag == 4:
			predictResults = predictResults[0]
		predictResults = predictResults.cpu().detach().numpy()
		#predictResults = predictResults.reshape(testLabels.shape)
	else:
		inp = net.input										   # input placeholder

		if BN_flag == 0 or BN_flag == 5:
			outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
			outputs = outputs[1:]
			last_conv_id = 10
		elif BN_flag == 1:
			outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
			outputs = outputs[1:]
			last_conv_id = 15
		elif BN_flag == 2:
			#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
			BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
			outputs = []
			for i in range(len(net.layers)):
				if i == 0 or i == 1:
					continue
				layer = net.layers[i]
				if i in BN_layer_ids:
					outputs.append(layer.get_output_at(0))
				else:
					outputs.append(layer.get_output_at(0))
					outputs.append(layer.get_output_at(1))
			last_conv_id = 22


		functor = K.function([inp], outputs )   # evaluation function
		
		### predict and output
		if BN_flag <= 1:
			layer_outs = functor(testData)
			predictResults = layer_outs[-1]
		else:
			layer_outs = functor([testData, testData])
			predictResults = layer_outs[-2+pathID]

		# print('[')
		# for c in range(layer_outs[9].shape[3]):
		# 	print(np.mean(layer_outs[9][:,:,:,c]), end=',')
		# print(']')

		# print('[')
		# for c in range(layer_outs[9].shape[3]):
		# 	print(np.std(layer_outs[9][:,:,:,c]), end=',')
		# print(']')

		# for j in range(len(testImagePaths)):
		# 	filename = os.path.basename(testImagePaths[j])
		# 	filename = filename.replace('.jpg', '.npy')
		# 	path = "../Data/udacityA_nvidiaB/trainC1_feat/" + filename
		# 	#print(layer_outs[9].shape)
		# 	np.save(path, layer_outs[9][j].transpose((2,0,1)))


		#predictResults = net.predict(testData)
		#score, acc = net.evaluate(testData, testLabels)

	predictResults = predictResults.flatten()


	if not pytorch_flag:
		BN_means = []
		BN_stds = []
		for layer_id in range(len(layer_outs)):
			#if layer_id not in [1, 3, 5, 7, 9]:
			#	continue

			layer_out = layer_outs[layer_id]
			#print(layer_out.shape)
			if layer_id <= last_conv_id:
				for i in range(layer_out.shape[3]):
					BN_means.append(np.mean(layer_out[:,:,:,i]))
					BN_stds.append(np.std(layer_out[:,:,:,i]))
			else:
				BN_means.append(np.mean(layer_out[:,:]))
				BN_stds.append(np.std(layer_out[:,:]))

		# print('mean ', np.mean(layer_outs[0]))
		# print('std ', np.std(layer_outs[0]))
		# adf

		if outputPath != "":
			f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_means.txt"),'w')
			#print(BN_means)
			#print(BN_stds)
			for mean in BN_means:
				f_BN.write("{:.5f}\n".format(mean))
			f_BN.close()
			f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_stds.txt"),'w')
			for std in BN_stds:
				f_BN.write("{:.5f}\n".format(std))
			f_BN.close()


	if outputPath != "":
		f = open(outputPath,'w')
	if fClassifier:
		predictResults = predictResults.argmax(axis=1)
		testLabels = testLabels.argmax(axis=1)
		correct_count = n - np.count_nonzero(predictResults-testLabels)
		print("accuracy: ", correct_count / (float)(n))

		if outputPath != "":
			f.write("accuracy: {:.5f}\n\n".format(correct_count / (float)(n)))
			f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
		
		for p in range(len(predictResults)):
	# 		if fClassifier:
	#  			f.write(str(np.argmax(p)))
	#  			print(np.argmax(p))
	# 		else: 
			# for regression
			imgName = os.path.basename(testImagePaths[p])
			prediction = predictResults[p]
			groundTruth = testLabels[p]

			if outputPath != "":
				f.write("{:^12.0f} {:^12.0f} {:^12.0f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
				f.write('\n')
	else:
		bin_metric(predictResults, testLabels)

		prediction_error = predictResults - testLabels
		mse_loss = np.mean(np.square(prediction_error))
		print("mse loss: " + str(mse_loss))

		if outputPath != "":
			f.write("mse loss: {:.5f}\n".format(mse_loss))

		#thresh_holds = [0.01, 0.033, 0.1, 0.33, 1, 3.3]
		thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
		#thresh_holds = [1, 2, 4, 8]
		acc_list = []

		for thresh_hold in thresh_holds:
			if outputPath != "":
				image_fail_cases_folder = os.path.dirname(outputPath)+'/fail_cases_'+str(thresh_hold)
				if not os.path.exists(image_fail_cases_folder):
					os.mkdir(image_fail_cases_folder)

			acc = np.sum(np.abs(prediction_error) < thresh_hold) / len(testLabels)
			acc_list.append(acc)
			print("accuracy (+-" + str(thresh_hold) + "): " + str(acc))

			if outputPath != "":
				f.write("accuracy (+-{:.3f}): {:.5f}\n".format(thresh_hold, acc))

				f_img_list_succ = open(outputPath.replace(ntpath.basename(outputPath), "img_list_"+str(thresh_hold)+"_succ_formal.txt"),'w')
				f_img_list_fail = open(outputPath.replace(ntpath.basename(outputPath), "img_list_"+str(thresh_hold)+"_fail_formal.txt"),'w')
				img_list_succ = testImagePaths[np.abs(prediction_error) < thresh_hold]
				label_succ = testLabels[np.abs(prediction_error) < thresh_hold]

				fail_flag = (np.abs(prediction_error) >= thresh_hold)

				#img_list_fail = testImagePaths[np.abs(prediction_error) >= thresh_hold]
				#img_fail = testData[np.abs(prediction_error) >= thresh_hold]
				#print(len(img_list_succ))
				#print(len(img_list_fail))
				for i in range(len(img_list_succ)):
					f_img_list_succ.write(os.path.basename(img_list_succ[i]) + ",,," + str(label_succ[i]) +"\n")
					#f_img_list_succ.write(str(img_file))
					#f_img_list_succ.write('\n')

				for i in range(len(fail_flag)):
					if fail_flag[i] == True:
						f_img_list_fail.write(os.path.basename(testImagePaths[i]) + ",,," + str(testLabels[i]) +"\n")
						#f_img_list_fail.write(str(img_file))
						#f_img_list_fail.write('\n')
						
						# img = cv2.imread(img_file)
						# #img_path = image_fail_cases_folder + "/gt_" + str(testLabels[i]) + "_pred_" + str(predictResults.flatten()[i]) + "_diff_" + str(prediction_error[i]) + "_" + os.path.basename(img_file)
						# img_path = image_fail_cases_folder + "/gt_" + "{:.3f}".format(testLabels[i]) + \
						# 	"_pred_" + "{:.3f}".format(predictResults.flatten()[i]) + \
						# 	"_diff_" + "{:.3f}".format(prediction_error[i]) + \
						# 	"_" + os.path.basename(img_file)
						# cv2.imwrite(img_path, img)
						


				f_img_list_succ.close()
				f_img_list_fail.close()

		print("mean accuracy: " + str(np.mean(acc_list)))
		if outputPath != "":
			f.write("mean accuracy: {:.5f} {:.2f}\n\n".format(np.mean(acc_list), np.mean(acc_list)*100))

		
			f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
			
			for p in range(len(predictResults)):
		# 		if fClassifier:
		#  			f.write(str(np.argmax(p)))
		#  			print(np.argmax(p))
		# 		else: 
				# for regression
				imgName = os.path.basename(testImagePaths[p])
				prediction = predictResults[p]
				groundTruth = testLabels[p]
				f.write("{:^12.3f} {:^12.3f} {:^12.3f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
				f.write('\n')

	if outputPath != "":
		f.close()

# 	for i in range(len(testLabels)):
# 		print([str('%.4f' % float(j)) for j in predictResults[i]])

			
	print('********************************************')
	print('\n\n\n')

	if (not pytorch_flag) and empty_net_flag:
		K.clear_session()

	return np.mean(acc_list)



def test_dnn_multi_pytorch(modelPath, imageDir_list, labelPath_list, outputPath, netType, flags, specs, BN_flag=0, pathID=0, ratio=1, pack_flag=False, pytorch_flag=False, net="", window_size_lstm=16, withFFT=False, Maxup_flag=True):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	specialFilter = 0
	if BN_flag == 7:
		specialFilter = 2
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	### retrieve the test data
	if not pack_flag:
		testImagePaths, testLabels = load_train_data_multi(imageDir_list, labelPath_list, nRep, fThreeCameras, ratio=ratio, specialFilter=specialFilter)
	else:
		testImagePaths, testLabels = load_train_data_multi_pack(imageDir_list, labelPath_list, nRep, fThreeCameras, ratio=ratio)

	#testImagePaths, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras, ratio=ratio)
	testImagePaths = np.array(testImagePaths)
	testLabels = np.array(testLabels)
	if pack_flag:
		testLabels = testLabels[:,0]
	n = len(testLabels)

	# testLabels = testLabels / 15



	# video_outpath = os.path.dirname(os.path.dirname(outputPath)) + "/video_outpath/"
	# if not os.path.exists(video_outpath):
	# 	os.mkdir(video_outpath)
	# video_file = video_outpath + os.path.basename(os.path.dirname(outputPath)) + ".mp4"

	# # out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 10, (455,256))  # for avi
	# # out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MP4V'), 5, (455,256))  # for mp4
	# out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MP4V'), 60, (1280,720))  # for mp4
	# # out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 10, (1280,720))  # for avi
	# alpha = 10
	# STEP = 1


	# for i,img_file in enumerate(testImagePaths):
	# 	if i % STEP != 0:
	# 		continue

	# 	# if ( i > 100 and i < 5000) or ( i > 5100 and i < 6000):
	# 	# if ( i > 100 and i < 5000) or ( i > 5100 and i < 6000):
	# 	# if i < 500 or ( i > 600 and i < 1500) or ( i > 1600 and i < 2500) or ( i > 2600 and i < 3500) or ( i > 3600 and i < 4500) or ( i > 4600 and i < 5500):
	# 		# continue

	# 	# if not ((i > 0 and i < 100) or (i > 4500 and i < 4600) or (i > 6000 and i < 6100)):
	# 	# if not ((i > 530 and i < 580) or (i > 2490 and i < 2540) ):

	# 	# if not ((i > 0 and i < 100) or (i > 1000 and i < 1100) or (i > 2000 and i < 2100) or (i > 3000 and i < 3100) or (i > 4000 and i < 4100) or (i > 5000 and i < 5100) or (i > 6000 and i < 6100)):
	# 	# if not ((i > 1200 and i < 1600)):
	# 	# if not ((i > 5000 and i < 5400)):
	# 	# if not ((i > 250 and i < 650) or (i > 1100 and i < 1500) or (i > 1600 and i < 2000)):
	# 	# 	continue


	# 	if not ((i > 0 and i < 400) or (i > 1400 and i < 1800) or (i > 2200 and i < 2600)):
	# 		continue

	# 	print(i)

	# 	img = cv2.imread(img_file)

	# 	# gt_v = testLabels[i]
	# 	# start_point = (227, 255)
	# 	# length = 80
	# 	# end_point = (int(start_point[0] + length*math.sin(gt_v*alpha/180*math.pi)), int(start_point[1] - length*math.cos(gt_v*alpha/180*math.pi)))
	# 	# img = cv2.line(img, start_point, end_point, (0, 255, 0), thickness=2) 

	# 	# predict_v = predictResults[i]
	# 	# start_point = (227, 255)
	# 	# length = 80
	# 	# end_point = (int(start_point[0] + length*math.sin(predict_v*alpha/180*math.pi)), int(start_point[1] - length*math.cos(predict_v*alpha/180*math.pi)))
	# 	# img = cv2.line(img, start_point, end_point, (0, 0, 255), thickness=2) 

	# 	# img = cv2.putText(img, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 

	# 	cv2.imshow("img", img)
	# 	cv2.waitKey(1)
	# 	out_video.write(img)

	# 	if i > 10000:
	# 		break

	# out_video.release()
	# return 0.5




	if fClassifier:
		print('\n######### Classification #########')
		testLabels = normalize_value(testLabels)
		testLabels = to_categorical(testLabels, num_classes = nClass)
	
	print(testImagePaths)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')

	size = (66, 200)
	if BN_flag == 8 or BN_flag == 10: #resnet
		size = (64, 64)

	transform=transforms.Compose([
		# transforms.Resize((66, 200)),
		transforms.Resize(size),
		transforms.ToTensor(),
		# transforms.Lambda(lambda rgb_img: 
		# 	torch.matmul(rgb_img.permute(1, 2, 0), 
		# 	yuv_weight.transpose(0, 1)).permute(2, 0, 1)),
		])

	test_dataset = DrivingDataset_pytorch(testImagePaths, testLabels, transform=transform, withFFT=withFFT)
	#dataset = DrivingDataset_pytorch(xTrainList, yTrainList)
	if platform == "win32":
		testGenerator = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
	else:
		testGenerator = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True)

	# if Maxup_flag:
	# 	testGenerator = gen_train_data_random(testImagePaths, testLabels, batchSize)

	# testData = []
	# if not pack_flag:
	# 	for i in range(len(testImagePaths)):
	# 		image_path = testImagePaths[i]
	# 		img = cv2.imread(image_path)
	# 		# if img == None:
	# 		# 	print(image_path)
	# 		img = resize_image(img)
	# 		testData.append(img)
	# else:
	# 	for i in range(len(testImagePaths)):
	# 		for j in range(len(testImagePaths[i])):
	# 			image_path = testImagePaths[i][j]
	# 			img_1 = cv2.imread(image_path)
	# 			# if img_1 == None:
	# 			# 	print(image_path)
	# 			img_1 = resize_image(img_1)
	# 			if j == 0:
	# 				img = img_1
	# 			else:
	# 				img = np.concatenate((img, img_1), axis=2)

	# 		#noise = np.random.uniform(low=0, high=255, size=(img.shape[0], img.shape[1], 1))
	# 		#img = np.concatenate((img, noise), axis=2)
	# 		testData.append(img)

	# testData = np.array(testData)

	nChannel=3
	if pack_flag:
		testImagePaths = testImagePaths[:,0]
		nChannel = 3*len(imageDir_list)

	# redistribution
	#testData = ((((testData / 127.5 - 1) - (-0.030626608) ) / 0.3224381 * 0.320617 + (-0.07968183)) + 1) * 127.5
	# print(np.min(testData))
	# print(np.max(testData))

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	empty_net_flag = False
	if net == "":
		empty_net_flag = True
		if pytorch_flag:
			if BN_flag == 9 or BN_flag == 10:
				net_f_ori, net_f_hint, net_d = create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel)
			else:
				net = create_nvidia_network_pytorch(BN_flag, fClassifier, nClass, nChannel, withFFT=withFFT)
		else:
			if netType == 1:
		# 		outputPath = trainPath + 'trainedModels/models-cnn/';
				net = create_nvidia_network(BN_flag, fClassifier, nClass, nChannel)
				if BN_flag == 3:
					net = net[0]
			elif netType == 2:
		# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
				net = net_lstm(2, nFramesSample)
			elif netType == 3:
		# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
				net = net_lstm(3, nFramesSample)

		## load model weights
		if modelPath != "":
			if pytorch_flag:
				if BN_flag == 9 or BN_flag == 10:
					modelPath1 = modelPath.replace("model-final", "model_f_ori-final")
					net_f_ori.load_state_dict(torch.load(modelPath1))
					modelPath1 = modelPath.replace("model-final", "model_f_hint-final")
					net_f_hint.load_state_dict(torch.load(modelPath1))
					modelPath1 = modelPath.replace("model-final", "model_d-final")
					net_d.load_state_dict(torch.load(modelPath1))

					net_f_ori.cuda()
					net_f_hint.cuda()
					net_d.cuda()

					net = net_f_ori
					net.eval()
					net.cuda()
				else:
					net.load_state_dict(torch.load(modelPath))
					net.eval()
					net.cuda()
			else:
				net.load_weights(modelPath)

	#print(net.layers[3].get_weights())
	if pytorch_flag:
		print(net)
	else:
		print(net.summary())


	# for name, param in net.named_parameters():
	# 	if param.requires_grad: 
	# 		print(name)

	# for param in net.parameters():
	# 	print(param.data)
	# 	adf

	net.eval()
	predictResults = []
	gtLabels = []
	netFeatures = []

	test_batch_num = int(len(testLabels)/batchSize)

	for i, (inputs, labels, _) in enumerate(testGenerator):
		# labels = labels.numpy().flatten()
		if BN_flag == 3:
			image, feature = inputs
			labels1, labels2 = labels
			# image = np.transpose(image, (0, 3, 1, 2))
			inputs = [image, feature]
			labels = np.concatenate((labels1, labels2))
			outputs = net(torch.Tensor(image).cuda(), torch.Tensor(feature).cuda(), mean2, std2)
		elif BN_flag == 7:
			if len(labels) < batchSize:
				break
			current_pos = random.randrange(window_size_lstm)
			inputs_seq = []
			labels_seq = []
			while current_pos + window_size_lstm <= batchSize:
				inputs_seq.append(inputs[current_pos:current_pos + window_size_lstm])
				labels_seq.append(labels[current_pos:current_pos + window_size_lstm])
				current_pos = current_pos + window_size_lstm

			inputs_seq = torch.stack(inputs_seq)
			labels = torch.stack(labels_seq)
			outputs,features = net(torch.Tensor(inputs_seq).cuda(non_blocking=True))
		elif BN_flag == 9 or BN_flag == 10: # HintNet
			# outputs,features = net_f_ori(torch.Tensor(inputs).cuda(non_blocking=True))
			features = net_f_ori(input_data=inputs.cuda(non_blocking=True))
			outputs,_ = net_d(features)

		else:
			# if Maxup_flag:
			# 	inputs = np.transpose(inputs, (0, 3, 1, 2))
			outputs,features = net(torch.Tensor(inputs).cuda(non_blocking=True))

		predictResults = predictResults + outputs.cpu().detach().numpy().flatten().tolist()


		if type(labels) is not np.ndarray:
			labels = labels.cpu().detach().numpy() # new pytorch version

		gtLabels = gtLabels + labels.flatten().tolist()
		netFeatures = netFeatures + features.cpu().detach().numpy().tolist()

		if i >= test_batch_num-1:
			break

	predictResults = np.array(predictResults).flatten()
	testLabels = np.array(gtLabels).flatten()
	netFeatures = np.array(netFeatures)

	if outputPath != "":
		f = open(outputPath,'w')
	if fClassifier:
		predictResults = predictResults.argmax(axis=1)
		testLabels = testLabels.argmax(axis=1)
		correct_count = n - np.count_nonzero(predictResults-testLabels)
		print("accuracy: ", correct_count / (float)(n))

		if outputPath != "":
			f.write("accuracy: {:.5f}\n\n".format(correct_count / (float)(n)))
			f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
		
		for p in range(len(predictResults)):
	# 		if fClassifier:
	#  			f.write(str(np.argmax(p)))
	#  			print(np.argmax(p))
	# 		else: 
			# for regression
			imgName = os.path.basename(testImagePaths[p])
			prediction = predictResults[p]
			groundTruth = testLabels[p]

			if outputPath != "":
				f.write("{:^12.0f} {:^12.0f} {:^12.0f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
				f.write('\n')
	else:
		bin_metric(predictResults, testLabels)

		prediction_error = predictResults - testLabels
		mse_loss = np.mean(np.square(prediction_error))
		print("mse loss: " + str(mse_loss))

		if outputPath != "":
			f.write("mse loss: {:.5f}\n".format(mse_loss))

		#thresh_holds = [0.01, 0.033, 0.1, 0.33, 1, 3.3]
		thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
		# thresh_holds = [1.5, 3, 7.5, 15, 30, 75]
		#thresh_holds = [1, 2, 4, 8]
		acc_list = []

		for thresh_hold in thresh_holds:
			if outputPath != "":
				image_fail_cases_folder = os.path.dirname(outputPath)+'/fail_cases_'+str(thresh_hold)
				if not os.path.exists(image_fail_cases_folder):
					os.mkdir(image_fail_cases_folder)

			acc = np.sum(np.abs(prediction_error) < thresh_hold) / len(testLabels)
			acc_list.append(acc)
			print("accuracy (+-" + str(thresh_hold) + "): " + str(acc))

			if outputPath != "":
				f.write("accuracy (+-{:.3f}): {:.5f}\n".format(thresh_hold, acc))

				# if BN_flag != 7:
				# 	label_name_clean = os.path.basename(labelPath_list[0]).replace('.csv', '')
				# 	f_img_list_succ = open(outputPath.replace(ntpath.basename(outputPath), label_name_clean+"_"+str(thresh_hold)+"_succ.csv"),'w')
				# 	f_img_list_fail = open(outputPath.replace(ntpath.basename(outputPath), label_name_clean+"_"+str(thresh_hold)+"_fail.csv"),'w')
				# 	img_list_succ = testImagePaths[np.abs(prediction_error) < thresh_hold]
				# 	label_succ = testLabels[np.abs(prediction_error) < thresh_hold]

				# 	fail_flag = (np.abs(prediction_error) >= thresh_hold)

				# 	#img_list_fail = testImagePaths[np.abs(prediction_error) >= thresh_hold]
				# 	#img_fail = testData[np.abs(prediction_error) >= thresh_hold]
				# 	#print(len(img_list_succ))
				# 	#print(len(img_list_fail))
				# 	for i in range(len(img_list_succ)):
				# 		f_img_list_succ.write(os.path.basename(img_list_succ[i]) + ",,," + str(label_succ[i]) +"\n")
				# 		#f_img_list_succ.write(str(img_file))
				# 		#f_img_list_succ.write('\n')

				# 	for i in range(len(fail_flag)):
				# 		if fail_flag[i] == True:
				# 			f_img_list_fail.write(os.path.basename(testImagePaths[i]) + ",,," + str(testLabels[i]) +"\n")
				# 			#f_img_list_fail.write(str(img_file))
				# 			#f_img_list_fail.write('\n')
							
				# 			# img = cv2.imread(testImagePaths[i])
				# 			# #img_path = image_fail_cases_folder + "/gt_" + str(testLabels[i]) + "_pred_" + str(predictResults.flatten()[i]) + "_diff_" + str(prediction_error[i]) + "_" + os.path.basename(img_file)
				# 			# img_path = image_fail_cases_folder + "/gt_" + "{:.3f}".format(testLabels[i]) + \
				# 			# 	"_pred_" + "{:.3f}".format(predictResults.flatten()[i]) + \
				# 			# 	"_diff_" + "{:.3f}".format(prediction_error[i]) + \
				# 			# 	"_" + os.path.basename(testImagePaths[i])
				# 			# cv2.imwrite(img_path, img)
							


				# 	f_img_list_succ.close()
				# 	f_img_list_fail.close()

		print("mean accuracy: " + str(np.mean(acc_list)))
		if outputPath != "":
			f.write("mean accuracy: {:.5f} {:.2f}\n\n".format(np.mean(acc_list), np.mean(acc_list)*100))

		
			f.write("{:^12} {:^12} {:^12} {:^12}\n".format("prediction", "groundtruth", "difference", "input"))
			
			for p in range(len(predictResults)):
		# 		if fClassifier:
		#  			f.write(str(np.argmax(p)))
		#  			print(np.argmax(p))
		# 		else: 
				# for regression
				imgName = os.path.basename(testImagePaths[p])
				prediction = predictResults[p]
				groundTruth = testLabels[p]
				f.write("{:^12.3f} {:^12.3f} {:^12.3f} {:^12}".format(prediction, groundTruth, prediction-groundTruth, imgName))
				f.write('\n')

			# if BN_flag == 7:
			# 	feature_path = outputPath.replace("test_result.txt", "lstm_feature.npy")
			# elif BN_flag == 8:
			# 	feature_path = outputPath.replace("test_result.txt", "resnet_feature.npy")
			# else:
			# 	feature_path = outputPath.replace("test_result.txt", "nvidiaNet_feature.npy")

			# np.save(feature_path, netFeatures)

	if outputPath != "":
		f.close()

# 	for i in range(len(testLabels)):
# 		print([str('%.4f' % float(j)) for j in predictResults[i]])

			
	print('********************************************')
	print('\n\n\n')

	if (not pytorch_flag) and empty_net_flag:
		K.clear_session()

	return np.mean(acc_list)





def test_dnn_visualize(modelPath, imageDir, labelPath, outputPath, netType, flags, specs, BN_flag=0, pathID=0, radius=5):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	step = 2
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	testImagePaths, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	testImagePaths = np.array(testImagePaths)
	testLabels = np.array(testLabels)
	n = len(testLabels)

	if fClassifier:
		print('\n######### Classification #########')
		testLabels = normalize_value(testLabels)
		testLabels = to_categorical(testLabels, num_classes = nClass)
	
	print(testImagePaths)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input										   # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
	
	for i in range(len(testLabels)):
		image_path = testImagePaths[i]
		label = testLabels[i]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")

		img_ori = cv2.imread(image_path)
		img = resize_image(img_ori)

		rows = (int)((img.shape[0]-1)/step) + 1
		cols = (int)((img.shape[1]-1)/step) + 1

		testData = []
		for i in range(rows):
			for j in range(cols):
				img1 = img.copy()
				st_r = np.clip(i*step-radius, 0, img.shape[0]-1)
				ed_r = np.clip(i*step+radius, 0, img.shape[0]-1)
				st_c = np.clip(j*step-radius, 0, img.shape[1]-1)
				ed_c = np.clip(j*step+radius, 0, img.shape[1]-1)
				img1[st_r:ed_r+1, st_c:ed_c+1, :] = 0
				testData.append(img1)
		testData.append(img)
		testData = np.array(testData)

		### predict and output
		if BN_flag <= 1:
			layer_outs = functor(testData)
			predictResults = layer_outs[-1]
		else:
			layer_outs = functor([testData, testData])
			predictResults = layer_outs[-2+pathID]
		#predictResults = net.predict(testData)
		#score, acc = net.evaluate(testData, testLabels)

		abs_diff = np.abs(predictResults.flatten() - label)
		max_diff = np.max(abs_diff)

		heat_map = np.reshape(np.abs(np.array(predictResults.flatten())[:(rows*cols)] - label), (rows, cols))
		heat_map = heat_map / np.max(heat_map)
		#cv2.imshow("heat_map_small", heat_map)

		heat_map = cv2.resize(heat_map,(img_ori.shape[1],img_ori.shape[0]), interpolation = cv2.INTER_AREA)
		#cv2.imshow("heat_map", heat_map)
		#cv2.imshow("img_ori", img_ori)

		img_ori[:,:,0] = np.multiply(img_ori[:,:,0], heat_map)
		img_ori[:,:,1] = np.multiply(img_ori[:,:,1], heat_map)
		img_ori[:,:,2] = np.multiply(img_ori[:,:,2], heat_map)
		#cv2.imshow("combine_img", img_ori)

		outputFolder = os.path.dirname(outputPath)
		imageName = os.path.basename(image_path)
		val_folder = imageName.replace(".jpg", "")
		val_folder = imageName.replace(".png", "")
		#outputImagePath = outputFolder + '/' + imageName + '_(gt_' + str(label) + ')_(error_' + str(abs_diff[-1]) + ')_(max_diff_' + str(max_diff) + ').jpg'
		outputImagePath = outputFolder + '/' + imageName + '_(gt_{:.3f})_(error_{:.3f})_(max_diff_{:.3f}).jpg'.format(label, abs_diff[-1], max_diff)

		cv2.imwrite(outputImagePath, img_ori)
		#cv2.waitKey(1);
		
			
	print('********************************************')
	print('\n\n\n')
	


def visualize_dnn_on_image(modelPath, imagePath, label, outputPath, netType, flags, specs, radius=10, BN_flag=0, pathID=0):
	
	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	img = resize_image(cv2.imread(imagePath))
	#img = cv2.imread(imagePath)
	#img = cv2.resize(img,(200, 66), interpolation = cv2.INTER_AREA)
	#cv2.imshow("test", img)
	#cv2.waitKey(0)

	step = 2

	rows = (int)((img.shape[0]-1)/step) + 1
	cols = (int)((img.shape[1]-1)/step) + 1

	testData = []
	for i in range(rows):
		for j in range(cols):
			img1 = img.copy()
			st_r = np.clip(i*step-radius, 0, img.shape[0]-1)
			ed_r = np.clip(i*step+radius, 0, img.shape[0]-1)
			st_c = np.clip(j*step-radius, 0, img.shape[1]-1)
			ed_c = np.clip(j*step+radius, 0, img.shape[1]-1)
			img1[st_r:ed_r+1, st_c:ed_c+1, :] = 0
			testData.append(img1)

	testData = np.array(testData)

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input										   # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
	
	### predict and output
	if BN_flag <= 1:
		layer_outs = functor(testData)
		predictResults = layer_outs[-1]
	else:
		layer_outs = functor([testData, testData])
		predictResults = layer_outs[-2+pathID]
	#predictResults = net.predict(testData)
	#score, acc = net.evaluate(testData, testLabels)

	BN_means = []
	BN_stds = []
	for layer_id in range(len(layer_outs)):
		#if layer_id not in [1, 3, 5, 7, 9]:
		#	continue

		layer_out = layer_outs[layer_id]
		#print(layer_out.shape)
		if layer_id <= last_conv_id:
			for i in range(layer_out.shape[3]):
				BN_means.append(np.mean(layer_out[:,:,:,i]))
				BN_stds.append(np.std(layer_out[:,:,:,i]))
		else:
			BN_means.append(np.mean(layer_out[:,:]))
			BN_stds.append(np.std(layer_out[:,:]))

			
	f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_means.txt"),'w')
	#print(BN_means)
	#print(BN_stds)
	for mean in BN_means:
		f_BN.write("{:.5f}\n".format(mean))
	f_BN.close()
	f_BN = open(outputPath.replace(ntpath.basename(outputPath), "BN_stds.txt"),'w')
	for std in BN_stds:
		f_BN.write("{:.5f}\n".format(std))
	f_BN.close()


	heat_map = np.reshape(np.abs(np.array(predictResults.flatten()) - label), (rows, cols))
	heat_map = heat_map / np.max(heat_map)
	cv2.imshow("heat_map_small", heat_map)

	img_ori = cv2.imread(imagePath)
	heat_map = cv2.resize(heat_map,(img_ori.shape[1],img_ori.shape[0]), interpolation = cv2.INTER_AREA)
	cv2.imshow("heat_map", heat_map)
	cv2.imshow("img_ori", img_ori)

	cv2.imshow("img_ori", img_ori)
	img_ori[:,:,0] = np.multiply(img_ori[:,:,0], heat_map)
	img_ori[:,:,1] = np.multiply(img_ori[:,:,1], heat_map)
	img_ori[:,:,2] = np.multiply(img_ori[:,:,2], heat_map)
	cv2.imshow("combine_img", img_ori)
	cv2.imwrite(outputPath, img_ori)
	cv2.waitKey(0);

# 	for i in range(len(testLabels)):
# 		print([str('%.4f' % float(j)) for j in predictResults[i]])

			
	print('********************************************')
	print('\n\n\n')



def read_float_list(file_name):
	x = []
	file_in = open(file_name, 'r')
	for y in file_in.read().split('\n'):
		if len(y) > 0:
			x.append(float(y))
	return x

def is_similar(val1, val2, val_thresh, percent_thresh):
	val_diff = abs(val1 - val2)
	percent_diff1 = abs(val_diff / val1)
	percent_diff2 = abs(val_diff / val2)

	if val_diff < val_thresh or (percent_diff1 < percent_thresh and percent_diff2 < percent_thresh):
		return True

	return False


def filter_dataset(modelPath, imageDir, labelPath, outputPath, netType, flags, specs, BN_flag=0, target_BN_folder="", filter_percent=0.1):
	

	## assigning variables
# 	fRandomDistort = flags[0]
	fThreeCameras  = flags[1]
	fClassifier	= flags[2]
# 	batchSize 	   = specs[0]
# 	nEpoch 		   = specs[1]
	nClass		 = specs[2]
	nFramesSample  = specs[3]
	nRep  = specs[4]
	
	print('\n\n\n')
	print('********************************************')
	
	if fClassifier:
		print('Classification......')
	else:
		print('Regression......')

	### retrieve the test data
	testImagePaths, testLabels = load_train_data(imageDir, labelPath, nRep, fThreeCameras)
	testImagePaths = np.array(testImagePaths)
	testLabels = np.array(testLabels)

	## choose networks, 1: CNN, 2: LSTM-m2o, 3: LSTM-m2m, 4: LSTM-o2o
	if netType == 1:
# 		outputPath = trainPath + 'trainedModels/models-cnn/';
		net = create_nvidia_network(BN_flag, fClassifier, nClass)
	elif netType == 2:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2o/'
		net = net_lstm(2, nFramesSample)
	elif netType == 3:
# 		outputPath = trainPath + 'trainedModels/models-lstm-m2m/'
		net = net_lstm(3, nFramesSample)

	#print(net.layers[3].get_weights())
	print(net.summary())
	
	## load model weights
	if modelPath != "":
		net.load_weights(modelPath)

	inp = net.input										   # input placeholder

	if BN_flag == 0:
		outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 10
	elif BN_flag == 1:
		outputs = [layer.get_output_at(-1) for layer in net.layers]		  # all layer outputs
		outputs = outputs[1:]
		last_conv_id = 15
	elif BN_flag == 2:
		#independent_layer_ids = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 24, 25, 28, 29, 32, 33]
		BN_layer_ids = [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 25, 26, 29, 30, 33, 34]
		outputs = []
		for i in range(len(net.layers)):
			if i == 0 or i == 1:
				continue
			layer = net.layers[i]
			if i in BN_layer_ids:
				outputs.append(layer.get_output_at(0))
			else:
				outputs.append(layer.get_output_at(0))
				outputs.append(layer.get_output_at(1))
		last_conv_id = 22


	functor = K.function([inp], outputs )   # evaluation function
		

	print(testImagePaths)
	print('The number of tested data: ' + str(testLabels.shape))
	print('********************************************')
	f = open(outputPath,'w')
	similar_img_count = 0
	diff_score_list = []
	for id in range(len(testLabels)):
		print(id)

		image_path = testImagePaths[id]
		label_value = testLabels[id]
		if not os.path.isfile(image_path):
			image_path = image_path.replace(".jpg", "_fake.png")
		img = resize_image(cv2.imread(image_path))
		testData=[img]

		testData = np.array(testData)

		### predict and output
		if BN_flag <= 1:
			layer_outs = functor(testData)
			predictResults = layer_outs[-1]
		else:
			layer_outs = functor([testData, testData])
			predictResults = layer_outs[-2]
		#predictResults = net.predict(testData)
		#score, acc = net.evaluate(testData, testLabels)


		BN_means = []
		BN_stds = []
		for layer_id in range(len(layer_outs)):
			#if layer_id not in [1, 3, 5, 7, 9]:
			#	continue

			layer_out = layer_outs[layer_id]
			if layer_id <= last_conv_id:
				for i in range(layer_out.shape[3]):
					BN_means.append(np.mean(layer_out[:,:,:,i]))
					BN_stds.append(np.std(layer_out[:,:,:,i]))
			else:
				BN_means.append(np.mean(layer_out[:,:]))
				BN_stds.append(np.std(layer_out[:,:]))

		BN_means_target = read_float_list(target_BN_folder + "/BN_means.txt")
		BN_stds_target = read_float_list(target_BN_folder + "/BN_stds.txt")

		SINGLE_FEATURE_PERCENT_THRESH = 1
		SINGLE_FEATURE_VALUE_THRESH = 3
		MODEL_PERCENT_THRESH = 0.5

		similar_feature_count = 0
		for i in range(len(BN_means)):
			mean_value_diff = abs(BN_means[i] - BN_means_target[i])
			mean_percent_diff = abs((BN_means[i] - BN_means_target[i])/BN_means_target[i])
			#print("base mean: ", BN_means_target[i], "  current mean: ", BN_means[i], "  mean_value_diff: ", mean_value_diff, "  mean_percent_diff: ", mean_percent_diff, "\n")
			
			std_value_diff = abs(BN_stds[i] - BN_stds_target[i])
			std_percent_diff = abs((BN_stds[i] - BN_stds_target[i])/BN_stds_target[i])
			#print("base std: ", BN_stds_target[i], "  current std: ", BN_stds[i], "  std_value_diff: ", std_value_diff, "  std_percent_diff: ", std_percent_diff, "\n")
			
			if is_similar(BN_means_target[i], BN_means[i], SINGLE_FEATURE_VALUE_THRESH, SINGLE_FEATURE_PERCENT_THRESH) and is_similar(BN_stds_target[i], BN_stds[i], SINGLE_FEATURE_VALUE_THRESH, SINGLE_FEATURE_PERCENT_THRESH):
				similar_feature_count += 1

		BN_means = np.array(BN_means)
		BN_means_target = np.array(BN_means_target)
		BN_stds = np.array(BN_stds)
		BN_stds_target = np.array(BN_stds_target)

		BN_mean_diff = BN_means - BN_means_target
		BN_std_diff = BN_stds - BN_stds_target

		'''
		mean_all_percent_1 = np.linalg.norm(BN_mean_diff) / np.linalg.norm(BN_means_target)
		mean_all_percent_2 = np.linalg.norm(BN_mean_diff) / np.linalg.norm(BN_means)
		std_all_percent_1 = np.linalg.norm(BN_std_diff) / np.linalg.norm(BN_stds_target)
		std_all_percent_2 = np.linalg.norm(BN_std_diff) / np.linalg.norm(BN_stds)

		#print("mean_all_percent_1 ", mean_all_percent_1, "mean_all_percent_2 ", mean_all_percent_2, "  std_all_percent_1 ", std_all_percent_1, "  std_all_percent_2 ", std_all_percent_2)
		
		# for honda
		MEAN_PERCENTAGE_THRESH = 1.2
		STD_PERCENTAGE_THRESH = 0.5

		# for udacity
		MEAN_PERCENTAGE_THRESH = 1.2
		STD_PERCENTAGE_THRESH = 0.8
		if mean_all_percent_1 < MEAN_PERCENTAGE_THRESH and mean_all_percent_2 < MEAN_PERCENTAGE_THRESH and std_all_percent_1 < MEAN_PERCENTAGE_THRESH and std_all_percent_2 < STD_PERCENTAGE_THRESH:
			f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
			similar_img_count += 1
		'''
		'''
		print(id, " ", similar_feature_count / len(BN_means))
		if similar_feature_count / len(BN_means) > MODEL_PERCENT_THRESH:
			f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
			similar_img_count += 1
		'''

		mean_std_ratio = np.linalg.norm(BN_means_target) / np.linalg.norm(BN_stds_target)
		diff_score = np.linalg.norm(BN_mean_diff) + mean_std_ratio * np.linalg.norm(BN_std_diff)
		#print("mean diff: ", np.linalg.norm(BN_mean_diff), "  std diff: ", np.linalg.norm(BN_std_diff), "  total: ", diff_score)
		#print("mean: ", np.linalg.norm(BN_means), "  std: ", np.linalg.norm(BN_stds))
		#diff_score_list.append((id, diff_score, np.linalg.norm(BN_means), np.linalg.norm(BN_stds)))
		diff_score_list.append((id, diff_score, np.linalg.norm(BN_mean_diff), np.linalg.norm(BN_std_diff)))

	f.close()
		
	#print(diff_score_list)
	diff_score_list = sorted(diff_score_list, key=lambda diff_score: diff_score[1])
	#print(diff_score_list)

	f = open(outputPath,'w')
	for i in range(int(len(testLabels)*filter_percent)):
		id = diff_score_list[i][0]
		image_path = testImagePaths[id]
		label_value = testLabels[id]
		#f.write(os.path.basename(image_path) + ",,," + str(label_value) + "\n")
		f.write(os.path.basename(image_path) + ",,," + str(label_value) + "," +str(diff_score_list[i][2]) + "," +str(diff_score_list[i][3])+"\n")
	f.close()

	print('similar_img_count ', int(len(testLabels)*filter_percent))
	print('total_count ', len(testLabels))
	print('ratio ', filter_percent)
	print('\n\n')

	
if __name__ == "__main__":
	print('\n')
	print("### This is the library file for testing. Please do not directly run it.")
	print('\n')
