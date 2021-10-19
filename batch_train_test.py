### This script is the main training file.

import sys
import os
import glob
import cv2
# cv2.setNumThreads(0)

ROOT_DIR = os.path.abspath("./")
print('PLATFORM_ROOT_DIR ', ROOT_DIR)

DATASET_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB/"
OUTPUT_ROOT = ROOT_DIR + "/Data/udacityA_nvidiaB_results/"
TRAIN_OUTPUT_ROOT = OUTPUT_ROOT + "train_results/"
TEST_OUTPUT_ROOT = OUTPUT_ROOT + "test_results/"

sys.path.insert(0, './library/')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from train import train_network, train_network_multi, train_network_multi_factor_search
from test import test_network, test_network_multi, visualize_network_on_image
from networks_pytorch import create_nvidia_network_pytorch
from networks import create_nvidia_network


def get_label_file_name(folder_name, suffix=""):
	pos = folder_name.find('_')
	if pos == -1:
		main_name = folder_name
	else:
		main_name = folder_name[0:pos]

	if "train" in folder_name:
		labelName = main_name.replace("train","labels") + "_train"
	elif "val" in folder_name:
		labelName = main_name.replace("val","labels") + "_val"

	labelName = labelName + suffix
	labelName = labelName + ".csv"
	return labelName


def single_test():
	train_folder = "trainWaymo"
	val_folder = "valWaymo"
	pytorch_flag = True

	imagePath = DATASET_ROOT + train_folder + "/"
	labelName = get_label_file_name(train_folder)
	# labelName = get_label_file_name("trainAudi3")
	labelPath = DATASET_ROOT + labelName

	outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
	train_network(imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag)

	if pytorch_flag:
		modelPath = outputPath + "/model-final.pth"
	else:
		modelPath = outputPath + "/model-final.h5"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder, "")
	labelPath = DATASET_ROOT + labelName
	#labelPath = DATASET_ROOT + "labelsB_train.csv"

	outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag)


def multi_factor_search(args):
	train_folder = args.train_folder
	val_folder = args.val_folder
	modelPath = args.model_path
	BN_flag = args.BN_flag
	train_suffix = args.suffix
	withFFT = args.with_FFT
	pytorch_flag = True

	if BN_flag > 0:
		train_suffix = train_suffix + "_BN" + str(BN_flag)
	if pytorch_flag:
		train_suffix = train_suffix + "_pytorch"
	if withFFT:
		train_suffix = train_suffix + "_withFFT"

	labelName = get_label_file_name(train_folder)
	labelPath = DATASET_ROOT + labelName
	imagePath = DATASET_ROOT + train_folder + "/"

	trainOurputFolder = train_folder + "_all_rob_20round_20epoch" + train_suffix
	# trainOurputFolder = train_folder + "_all_rob_20epoch_multi_retrain"
	trainOutputPath = TRAIN_OUTPUT_ROOT + trainOurputFolder + "/"
	#modelPath = TRAIN_OUTPUT_ROOT + "trainB_quality_channelGSY/model-final.h5"
	train_network_multi_factor_search(imagePath, labelPath, trainOutputPath, modelPath=modelPath, BN_flag=BN_flag, pytorch_flag=pytorch_flag, withFFT=withFFT)

	modelPath = trainOutputPath + "/model-final.h5"
	if pytorch_flag:
		modelPath = trainOutputPath + "/model-final.pth"

	imagePath = DATASET_ROOT + val_folder + "/"
	labelName = get_label_file_name(val_folder, "")
	labelPath = DATASET_ROOT + labelName
	#labelPath = DATASET_ROOT + "labelsB_train.csv"

	valOutputPath = TEST_OUTPUT_ROOT + "(" + trainOurputFolder + ")_(" + val_folder + ")/test_result.txt"
	#modelPath = ""
	test_network(modelPath, imagePath, labelPath, valOutputPath, BN_flag=BN_flag, pytorch_flag=pytorch_flag)


def reformal_corruption_accs(MA_list):
	corruption_accs_1 = [MA_list[0]]
	corruption_accs_2 = [MA_list[1:6], MA_list[6:11], MA_list[11:16], MA_list[16:26], MA_list[26:36], MA_list[36:46], MA_list[46:56], MA_list[56:66], MA_list[66:76]]
	corruption_accs_3 = [MA_list[76:77], MA_list[77:78], MA_list[78:79], MA_list[79:80], MA_list[80:81], MA_list[81:82]]
	corruption_accs_4 = [MA_list[82:87], MA_list[87:92], MA_list[92:97], MA_list[97:102], MA_list[102:107], MA_list[107:112], MA_list[112:117]]
	return corruption_accs_1, corruption_accs_2, corruption_accs_3, corruption_accs_4


def compute_mce(corruption_accs, base_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  n = len(corruption_accs)
  for i in range(n):
    avg_err = 1 - np.mean(corruption_accs[i])
    base_err = 1 - np.mean(base_accs[i])
    ce = 100 * avg_err / base_err
    mce += ce / n
  return mce


def unit_test_for_robust():
	TRAIN_LIST = ["trainB_all_rob_20epoch_pytorch_withFFT"]
	# TRAIN_LIST = ["trainHc_all_rob_20epoch_pytorch_withFFT"]
	# TRAIN_LIST = ["trainAds_all_rob_20epoch_pytorch_withFFT"]

	# # TRAIN_LIST = ["trainAudi6_trainAudi6segall_hintnet_1_2000r_1e"]
	# TRAIN_LIST = ["trainAudi6_trainAudi6segall_hintnet_r002_t02_2000r_1e_uniform_200pre"]
	# TRAIN_LIST = ["trainAudi6"]
	# TRAIN_LIST = ["trainAudi6_trainAudi6seginfer_hintnet_100Ia_2000r_1e"]

	# TRAIN_LIST = ["trainHonda100k"]
	# TRAIN_LIST = ["trainHonda100k_augmix"]
	# TRAIN_LIST = ["trainHonda100k_all_rob_20round_5epoch_pytorch_withFFT"]
	# TRAIN_LIST = ["trainHonda100k_all_rob_20round_5epoch_pytorch"]
	# TRAIN_LIST = ["trainHonda100k_pytorch_retrain"]

	TRAIN_LIST = ["trainWaymo_all_rob_50round_2epoch_pytorch"]
	TRAIN_LIST = ["trainWaymo"]
	TRAIN_LIST = ["trainWaymo_augmix"]
	TRAIN_LIST = ["trainWaymo_all_rob_50round_2epoch_pytorch_withFFT"]

	VAL_LIST = ["valB", \
				"valB_blur_1", "valB_blur_2", "valB_blur_3", "valB_blur_4", "valB_blur_5", \
				"valB_noise_1", "valB_noise_2", "valB_noise_3", "valB_noise_4", "valB_noise_5", \
				"valB_distort_1", "valB_distort_2", "valB_distort_3", "valB_distort_4", "valB_distort_5", \
				"valB_R_darker_1", "valB_R_darker_2", "valB_R_darker_3", "valB_R_darker_4", "valB_R_darker_5", \
				"valB_R_lighter_1", "valB_R_lighter_2", "valB_R_lighter_3", "valB_R_lighter_4", "valB_R_lighter_5", \
				"valB_G_darker_1", "valB_G_darker_2", "valB_G_darker_3", "valB_G_darker_4", "valB_G_darker_5", \
				"valB_G_lighter_1", "valB_G_lighter_2", "valB_G_lighter_3", "valB_G_lighter_4", "valB_G_lighter_5", \
				"valB_B_darker_1", "valB_B_darker_2", "valB_B_darker_3", "valB_B_darker_4", "valB_B_darker_5", \
				"valB_B_lighter_1", "valB_B_lighter_2", "valB_B_lighter_3", "valB_B_lighter_4", "valB_B_lighter_5", \
				"valB_H_darker_1", "valB_H_darker_2", "valB_H_darker_3", "valB_H_darker_4", "valB_H_darker_5", \
				"valB_H_lighter_1", "valB_H_lighter_2", "valB_H_lighter_3", "valB_H_lighter_4", "valB_H_lighter_5", \
				"valB_S_darker_1", "valB_S_darker_2", "valB_S_darker_3", "valB_S_darker_4", "valB_S_darker_5", \
				"valB_S_lighter_1", "valB_S_lighter_2", "valB_S_lighter_3", "valB_S_lighter_4", "valB_S_lighter_5", \
				"valB_V_darker_1", "valB_V_darker_2", "valB_V_darker_3", "valB_V_darker_4", "valB_V_darker_5", \
				"valB_V_lighter_1", "valB_V_lighter_2", "valB_V_lighter_3", "valB_V_lighter_4", "valB_V_lighter_5", \
				# "valB_combined_1_3", "valB_combined_1_4", "valB_combined_1_7", \
				# "valB_combined_1_8", "valB_combined_1_9", "valB_combined_1_10", \
				"valB_combined_1_0", "valB_combined_2_0", "valB_combined_3_0", \
				"valB_combined_4_0", "valB_combined_5_0", "valB_combined_6_0", \
				"valB_IMGC_motion_blur_1", "valB_IMGC_motion_blur_2", "valB_IMGC_motion_blur_3", \
				"valB_IMGC_motion_blur_4", "valB_IMGC_motion_blur_5", \
				"valB_IMGC_zoom_blur_1", "valB_IMGC_zoom_blur_2", "valB_IMGC_zoom_blur_3", \
				"valB_IMGC_zoom_blur_4", "valB_IMGC_zoom_blur_5", \
				"valB_IMGC_pixelate_1", "valB_IMGC_pixelate_2", "valB_IMGC_pixelate_3", \
				"valB_IMGC_pixelate_4", "valB_IMGC_pixelate_5", \
				"valB_IMGC_jpeg_compression_1", "valB_IMGC_jpeg_compression_2", "valB_IMGC_jpeg_compression_3", \
				"valB_IMGC_jpeg_compression_4", "valB_IMGC_jpeg_compression_5", \
				"valB_IMGC_snow_1", "valB_IMGC_snow_2", "valB_IMGC_snow_3", \
				"valB_IMGC_snow_4", "valB_IMGC_snow_5", \
				"valB_IMGC_frost_1", "valB_IMGC_frost_2", "valB_IMGC_frost_3", \
				"valB_IMGC_frost_4", "valB_IMGC_frost_5", \
				"valB_IMGC_fog_1", "valB_IMGC_fog_2", "valB_IMGC_fog_3", \
				"valB_IMGC_fog_4", "valB_IMGC_fog_5"
				]

	MA_list = []
	for train_folder in TRAIN_LIST:
		imagePath = DATASET_ROOT + train_folder + "/"
		labelName = get_label_file_name(train_folder)
		labelPath = DATASET_ROOT + labelName
		outputPath = TRAIN_OUTPUT_ROOT + train_folder + "/"
		if "_pytorch" in train_folder:
			pytorch_flag = True
		else:
			pytorch_flag = False
		pytorch_flag = True

		withFFT = False
		if "_withFFT" in train_folder:
			withFFT = True

		#train_network(imagePath, labelPath, outputPath, pytorch_flag=pytorch_flag)

		modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.h5"
		if pytorch_flag:
			modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model-final.pth"
			# modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/checkpoint_999.pth"
			# modelPath = TRAIN_OUTPUT_ROOT + train_folder + "/model_best.pth"

		net=""
		BN_flag = 0
		if "commaai" in train_folder:
			BN_flag = 5
		if ("resnet" in train_folder) or ("_BN8" in train_folder):
			BN_flag = 8
		if ("hintnet" in train_folder):
			BN_flag = 9


		if modelPath != "" and BN_flag!=9:
			if pytorch_flag:
				net = create_nvidia_network_pytorch(BN_flag, withFFT=withFFT)
				if "augmix" in train_folder:
					net = torch.nn.DataParallel(net).cuda()
					net.load_state_dict(torch.load(modelPath)['state_dict'])
				else:
					net.load_state_dict(torch.load(modelPath))
					net.cuda()
				net.eval()
			else:
				net = create_nvidia_network(0, False, 0, 3)
				net.load_weights(modelPath)

		ratio = 1
		if "trainHc" in train_folder:
			for i in range(len(VAL_LIST)):
				VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valHc')
		elif "trainAds" in train_folder:
			for i in range(len(VAL_LIST)):
				VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valAds')
		elif "trainAudi6" in train_folder:
			for i in range(len(VAL_LIST)):
				VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valAudi6')
		elif "trainHonda100k" in train_folder:
			ratio = 0.1
			for i in range(len(VAL_LIST)):
				VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valHonda100k')
		elif "trainWaymo" in train_folder:
			ratio = 0.1
			for i in range(len(VAL_LIST)):
				VAL_LIST[i] = VAL_LIST[i].replace('valB', 'valWaymo')
				# VAL_LIST[i] = VAL_LIST[i].replace('valB', 'trainWaymo')



		for val_folder in VAL_LIST:
			# val_folder = val_folder.replace("train", "val")

			#if not (train_folder == "trainA_MUNIT_GAN_1" or val_folder == "valA_MUNIT_GAN"):
			#	continue

			imagePath = DATASET_ROOT + val_folder + "/"
			labelName = get_label_file_name(val_folder)
			labelPath = DATASET_ROOT + labelName
			outputPath = TEST_OUTPUT_ROOT + "(" + train_folder + ")_(" + val_folder + ")/test_result.txt"
			MA=0
			MA = test_network(modelPath, imagePath, labelPath, outputPath, BN_flag=BN_flag, pytorch_flag=pytorch_flag, net=net, withFFT=withFFT, ratio=ratio) #, visualize=True
			MA_list.append(MA)

		MA_list = np.array(MA_list) * 100

		# print(MA_list[0])
		# for i in range(int(len(MA_list)/5)):
		# 	print(np.mean(MA_list[i*5+1:(i+1)*5+1]))

		#TODO
		# MA_list = [74.20, 74.29, 74.49, 74.82, 75.09, 76.28, 74.26, 74.79, 74.46, 74.97, 76.19, 74.14, 75.09, 65.83, 54.76, 57.11, 74.23, 73.96, 73.99, 74.05, 74.08, 74.26, 75.03, 74.35, 74.85, 76.73, 74.17, 74.14, 74.40, 74.46, 74.94, 74.23, 74.73, 75.12, 75.71, 75.33, 74.20, 74.02, 73.93, 73.99, 74.29, 74.20, 74.85, 74.67, 74.82, 75.80, 74.40, 74.43, 74.35, 74.17, 73.72, 74.26, 73.90, 74.11, 74.23, 73.69, 74.23, 74.61, 74.55, 74.43, 74.23, 74.20, 73.99, 74.40, 74.55, 74.20, 74.32, 74.26, 74.64, 74.52, 75.00, 74.14, 74.46, 74.05, 76.04, 82.56, 69.88, 74.85, 70.86, 74.61, 74.23, 69.49, 74.29, 74.46, 74.91, 75.06, 74.76, 73.66, 73.24, 73.36, 72.95, 72.35, 74.17, 74.23, 74.23, 74.26, 74.32, 74.35, 74.05, 74.20, 74.11, 74.23, 74.64, 75.12, 75.03, 73.96, 71.55, 75.24, 75.71, 75.77, 75.98, 76.40, 74.70, 74.94, 73.60, 72.86, 70.74]
		# MA_list = [80.57, 80.74, 80.48, 79.82, 78.1, 73.21, 80.27, 80.39, 79.46, 77.59, 76.46, 80.03, 77.77, 75.3, 72.95, 43.72, 80.54, 79.94, 79.11, 78.99, 78.63, 80.6, 80.21, 79.38, 79.29, 77.83, 80.51, 80, 78.69, 78.51, 78.36, 80.39, 79.32, 78.04, 77.44, 78.27, 80.39, 79.76, 78.84, 78.81, 78.54, 80.6, 80.45, 79.7, 79.35, 78.42, 80.57, 79.82, 79.64, 78.87, 79.7, 80.57, 79.35, 79.43, 78.21, 79.73, 80.6, 80.57, 80.18, 79.43, 77.98, 80.57, 80.18, 79.11, 79.11, 76.96, 80.27, 78.45, 76.55, 76.31, 76.55, 80.71, 79.14, 77.14, 74.94, 67.77, 70.62, 64.64, 67.83, 75.89, 74.49, 57.62, 79.32, 78.63, 77.41, 74.91, 72.17, 78.96, 78.21, 77.38, 76.34, 74.88, 80.68, 80.65, 80.48, 80.68, 80.42, 80.54, 80.74, 80.27, 79.97, 79.85, 77.89, 53.33, 63.33, 70.62, 68.33, 76.37, 75.57, 74.05, 74.61, 73.93, 73.75, 71.31, 66.61, 62.83, 57.32]
		# MA_list = np.array(MA_list)
		
		# base_MA_resnet_Hc = [82.20, 82.17, 81.52, 77.12, 71.86, 68.35, 82.15, 80.59, 75.46, 70.70, 67.80, 80.30, 76.37, 71.30, 66.95, 68.51, 82.05, 78.40, 70.93, 67.58, 62.02, 82.03, 77.71, 71.46, 70.46, 70.43, 82.12, 82.13, 82.03, 81.37, 80.24, 82.08, 80.92, 76.69, 74.79, 69.78, 82.10, 81.90, 78.45, 75.89, 72.19, 82.07, 77.19, 67.70, 62.50, 57.03, 82.03, 81.77, 81.55, 79.64, 77.56, 82.23, 81.65, 78.62, 77.97, 77.59, 82.10, 81.59, 80.10, 79.72, 77.91, 82.20, 80.29, 74.33, 72.18, 66.75, 82.10, 79.14, 71.28, 62.59, 58.34, 82.18, 75.54, 64.34, 64.10, 63.09, 70.26, 63.32, 69.95, 72.26, 65.15, 62.50, 82.07, 81.92, 80.77, 78.94, 77.66, 81.05, 80.54, 79.92, 79.32, 78.60, 82.17, 82.25, 82.17, 81.92, 81.83, 82.18, 82.10, 82.25, 81.95, 82.03, 74.04, 68.96, 68.33, 65.50, 64.14, 69.18, 64.60, 62.69, 62.84, 61.99, 64.97, 61.75, 57.91, 56.14, 52.91]

		# base_MA_nvidia_B = [88.36, 88.21, 88.06, 86.07, 81.16, 73.33, 88.33, 86.01, 81.42, 76.39, 73.15, 88.63, 74.97, 57.67, 48.83, 49.16, 87.81, 65.06, 57.91, 55.44, 53.24, 87.67, 61.36, 52.14, 47.44, 45.14, 88.54, 73, 53.54, 48.21, 44.16, 87.91, 69.55, 51.16, 43.66, 39.97, 88.18, 69.73, 54.25, 46.84, 43.03, 87.7, 66.22, 52.47, 47.14, 42.58, 88.09, 82.76, 63.06, 52.14, 51.28, 88.15, 69.25, 51.51, 51.33, 51.22, 88.06, 83.89, 72.56, 63.83, 58.36, 88.33, 74.46, 61.57, 56.48, 53.15, 88.51, 69.43, 54.61, 53.21, 52.63, 88.36, 70.35, 49.1, 43.21, 39.43, 59.73, 54.02, 40.89, 50.06, 54.02, 56.31, 76.4, 69.7, 62.62, 61.1, 60.33, 85.57, 83.66, 81.79, 79.97, 78.15, 88.15, 88.21, 88.04, 88.27, 88.1, 88.42, 88.01, 87.41, 85.39, 82.17, 62.77, 50.68, 54.94, 55.45, 55.33, 55.8, 52.14, 51.67, 51.67, 51.22, 58.72, 55, 52.44, 50.8, 48.12]
		# base_MA = np.array(base_MA_nvidia_B)
		

		# MA_list = [89.29, 89.2, 89.46, 88.75, 82.36, 75.45, 89.08, 88.66, 88.54, 85.54, 82.74, 89.14, 85.51, 63.1, 56.49, 50.6, 89.4, 89.49, 89.35, 88.81, 87.29, 89.35, 89.35, 89.7, 89.11, 87.44, 89.4, 89.55, 89.67, 89.32, 88.39, 89.29, 89.46, 89.7, 89.38, 88.54, 89.38, 89.2, 89.49, 89.49, 88.99, 89.29, 89.4, 89.52, 89.29, 88.9, 89.2, 88.48, 89.05, 88.36, 88.69, 89.17, 89.14, 88.36, 87.8, 88.66, 89.26, 89.02, 88.24, 87.77, 85.65, 89.29, 89.32, 88.54, 88.18, 84.52, 89.32, 89.73, 86.76, 80.6, 61.88, 89.29, 89.08, 81.37, 74.49, 77.71, 71.28, 61.13, 65.6, 83.3, 85.62, 54.52, 76.04, 68.07, 59.4, 57.92, 58.12, 87.38, 85.83, 83.6, 81.76, 79.88, 89.61, 89.67, 89.61, 89.61, 89.52, 89.49, 89.52, 89.64, 89.23, 89.38, 86.9, 56.16, 66.88, 75.8, 74.64, 84.91, 81.46, 79.17, 79.26, 77.59, 77.62, 73.15, 67.17, 63.42, 57.86]
		# MA_list = np.array(MA_list)


		MA_list_nvidia_Hc_augmix = [75.72, 75.67, 75.79, 74.89, 73.26, 68.15, 75.61, 75.64, 75.62, 75.49, 75.06, 75.37, 72.84, 69.41, 66.22, 67.25, 75.67, 75.64, 75.72, 75.66, 75.81, 75.57, 75.34, 75.67, 75.51, 75.54, 75.71, 75.59, 75.42, 75.49, 75.46, 75.71, 75.46, 75.71, 75.67, 75.67, 75.66, 75.69, 75.34, 75.36, 75.37, 75.69, 75.62, 75.67, 75.41, 74.99, 75.74, 75.54, 75.69, 75.51, 75.67, 75.57, 75.57, 75.47, 75.54, 75.59, 75.56, 75.41, 75.46, 75.37, 75.27, 75.71, 75.81, 75.79, 75.51, 75.24, 75.71, 75.61, 75.46, 75.51, 74.53, 75.66, 75.51, 74.66, 73.36, 65.47, 73.44, 70.05, 69.20, 74.56, 75.26, 70.88, 75.21, 74.76, 73.86, 72.93, 71.55, 74.96, 74.64, 74.43, 74.06, 73.33, 75.87, 75.57, 75.59, 75.62, 75.81, 75.62, 75.87, 75.79, 75.72, 75.32, 74.99, 74.09, 74.11, 72.69, 73.94, 74.74, 74.29, 73.89, 74.24, 73.66, 75.64, 75.37, 74.43, 73.48, 71.38]
		MA_list_nvidia_Hc_ours = [78.32, 78.24, 78.32, 78.35, 76.96, 75.02, 78.29, 78.50, 78.39, 77.82, 76.44, 77.77, 74.44, 75.46, 76.04, 70.01, 78.40, 78.45, 78.64, 78.45, 78.77, 78.47, 78.60, 78.55, 78.24, 78.26, 78.40, 78.55, 78.65, 78.65, 78.45, 78.44, 78.62, 78.55, 78.47, 78.54, 78.37, 78.37, 78.40, 78.32, 78.04, 78.40, 78.60, 78.29, 78.27, 77.89, 78.44, 78.39, 78.44, 78.49, 78.29, 78.44, 78.26, 78.39, 78.35, 78.31, 78.45, 78.54, 78.21, 78.24, 77.71, 78.47, 78.49, 78.55, 78.22, 77.54, 78.42, 78.74, 78.04, 76.91, 58.99, 78.45, 77.99, 76.64, 75.19, 69.78, 72.41, 71.76, 75.02, 76.91, 78.21, 66.20, 78.27, 77.41, 76.64, 74.23, 72.86, 77.29, 76.76, 76.74, 76.27, 75.67, 78.31, 78.24, 78.44, 78.39, 78.31, 78.26, 78.47, 78.26, 77.87, 78.27, 76.89, 75.77, 74.91, 71.51, 73.94, 76.61, 75.52, 74.59, 74.51, 73.91, 75.46, 72.78, 67.37, 64.20, 59.27]
		
		MA_list_commaai_Hc_augmix = [68.98, 69.50, 68.93, 69.28, 66.18, 63.20, 69.65, 69.21, 67.83, 67.13, 66.22, 68.41, 59.27, 67.23, 66.97, 67.30, 69.63, 69.55, 68.85, 69.36, 68.68, 69.95, 68.73, 68.55, 68.75, 68.48, 69.60, 68.35, 68.40, 68.80, 67.75, 69.48, 69.71, 68.53, 68.53, 67.55, 68.50, 68.51, 68.53, 68.81, 68.08, 69.11, 68.51, 69.00, 67.88, 65.90, 69.41, 68.13, 69.25, 68.80, 69.06, 69.38, 69.03, 68.41, 69.11, 68.40, 68.05, 68.71, 69.46, 68.78, 68.81, 69.51, 69.11, 68.17, 68.17, 65.63, 69.40, 68.38, 67.52, 66.18, 58.08, 68.61, 68.15, 67.50, 65.85, 55.89, 53.10, 67.00, 67.88, 63.59, 68.02, 61.31, 69.28, 68.15, 68.46, 67.12, 65.77, 68.43, 67.17, 68.50, 67.72, 67.33, 68.96, 69.25, 69.00, 68.91, 68.12, 68.98, 68.58, 70.26, 69.38, 69.30, 68.78, 67.60, 65.58, 63.14, 65.88, 68.41, 67.38, 65.70, 65.30, 64.87, 65.92, 66.17, 63.25, 62.22, 57.84]
		MA_list_commaai_Hc_ours = []

		MA_list_nvidia_Ads_augmix = [82.84, 82.66, 81.89, 79.86, 71.15, 59.03, 82.66, 82.05, 81.96, 80.18, 77.85, 82.26, 70.73, 54.37, 17.32, 11.76, 82.96, 83.22, 82.00, 81.16, 78.83, 83.08, 83.10, 82.24, 81.49, 79.46, 83.03, 82.61, 82.35, 82.12, 79.55, 83.03, 83.19, 82.33, 80.91, 76.98, 82.98, 82.82, 82.59, 81.68, 78.76, 83.03, 83.38, 82.73, 82.21, 80.72, 83.19, 82.70, 82.52, 83.03, 81.84, 83.05, 82.33, 82.84, 83.01, 81.84, 83.12, 83.47, 82.31, 81.47, 79.34, 83.15, 83.31, 79.90, 77.31, 66.39, 83.01, 83.10, 82.26, 81.09, 14.01, 82.89, 82.14, 79.06, 75.58, 53.80, 70.68, 52.31, 53.62, 77.61, 77.71, 54.86, 81.56, 79.62, 75.70, 71.52, 68.02, 79.65, 77.38, 75.56, 73.95, 71.36, 82.61, 82.66, 82.28, 82.49, 82.40, 82.75, 82.47, 82.70, 81.98, 81.98, 81.07, 79.34, 77.22, 73.97, 75.63, 80.67, 78.45, 75.98, 76.17, 74.63, 80.23, 78.66, 73.20, 66.62, 54.67]
		MA_list_nvidia_Ads_ours = [95.21, 95.38, 95.10, 93.98, 91.29, 94.07, 95.42, 95.00, 94.51, 92.27, 90.71, 93.07, 73.51, 89.03, 51.49, 46.43, 95.99, 95.89, 95.45, 95.45, 95.12, 95.99, 95.80, 95.38, 95.10, 94.07, 96.03, 96.13, 96.06, 96.20, 95.28, 95.94, 96.06, 95.96, 96.15, 95.68, 95.99, 96.22, 96.17, 95.92, 94.14, 96.03, 96.13, 96.24, 96.17, 95.35, 95.77, 95.77, 95.70, 95.26, 95.31, 95.87, 95.63, 95.70, 94.65, 95.31, 96.01, 95.54, 94.07, 93.37, 91.18, 96.01, 95.68, 93.88, 93.28, 91.18, 96.15, 96.10, 92.02, 71.45, 53.06, 96.03, 95.35, 88.91, 81.58, 50.37, 64.12, 65.27, 84.97, 82.54, 94.37, 49.98, 92.27, 89.08, 83.85, 76.47, 71.15, 87.77, 84.36, 81.37, 77.94, 73.88, 95.31, 95.28, 94.98, 95.10, 95.07, 95.05, 94.93, 94.93, 94.56, 93.51, 91.57, 86.97, 83.50, 78.62, 76.05, 88.40, 85.81, 82.31, 81.96, 79.41, 72.67, 63.00, 49.84, 43.00, 33.36]

		MA_list_commaai_Ads_augmix = [60.64, 60.13, 60.18, 59.57, 53.78, 46.85, 59.71, 59.17, 57.98, 58.17, 56.47, 57.49, 37.63, 37.84, 34.06, 8.73, 60.29, 59.76, 60.04, 58.96, 52.08, 59.38, 58.68, 54.23, 51.54, 43.79, 59.13, 59.06, 55.49, 50.09, 37.30, 59.85, 59.69, 56.47, 52.08, 42.18, 59.20, 58.68, 58.10, 56.19, 52.64, 60.13, 58.43, 58.43, 56.82, 49.25, 60.67, 60.18, 58.68, 58.22, 59.15, 60.13, 58.52, 59.43, 58.87, 60.29, 59.76, 59.52, 58.66, 58.92, 57.31, 60.01, 59.92, 58.96, 57.59, 45.92, 60.57, 61.20, 58.96, 54.65, 34.10, 59.99, 57.87, 52.24, 45.82, 32.19, 38.28, 35.22, 37.61, 38.49, 55.67, 20.05, 57.66, 59.36, 56.16, 55.23, 53.76, 58.12, 56.40, 54.32, 53.66, 51.94, 59.34, 59.24, 59.24, 59.36, 60.39, 59.83, 59.31, 59.78, 58.29, 58.45, 54.30, 49.11, 46.83, 43.88, 44.07, 51.73, 49.63, 47.29, 47.53, 46.78, 48.39, 43.07, 38.68, 37.58, 32.40]
		MA_list_commaai_Ads_ours = []

		base_MA_nvidia_B = [88.36, 88.21, 88.06, 86.07, 81.16, 73.33, 88.33, 86.01, 81.42, 76.39, 73.15, 88.63, 74.97, 57.67, 48.83, 49.16, 87.81, 65.06, 57.91, 55.44, 53.24, 87.67, 61.36, 52.14, 47.44, 45.14, 88.54, 73, 53.54, 48.21, 44.16, 87.91, 69.55, 51.16, 43.66, 39.97, 88.18, 69.73, 54.25, 46.84, 43.03, 87.7, 66.22, 52.47, 47.14, 42.58, 88.09, 82.76, 63.06, 52.14, 51.28, 88.15, 69.25, 51.51, 51.33, 51.22, 88.06, 83.89, 72.56, 63.83, 58.36, 88.33, 74.46, 61.57, 56.48, 53.15, 88.51, 69.43, 54.61, 53.21, 52.63, 88.36, 70.35, 49.1, 43.21, 39.43, 59.73, 54.02, 40.89, 50.06, 54.02, 56.31, 76.4, 69.7, 62.62, 61.1, 60.33, 85.57, 83.66, 81.79, 79.97, 78.15, 88.15, 88.21, 88.04, 88.27, 88.1, 88.42, 88.01, 87.41, 85.39, 82.17, 62.77, 50.68, 54.94, 55.45, 55.33, 55.8, 52.14, 51.67, 51.67, 51.22, 58.72, 55, 52.44, 50.8, 48.12]
		base_MA_nvidia_Hc = [75.84, 75.69, 75.44, 74.51, 70.05, 61.99, 75.87, 75.36, 74.83, 72.26, 69.81, 74.91, 69.06, 66.80, 71.50, 62.79, 75.77, 69.85, 58.47, 53.33, 46.47, 75.79, 66.73, 54.43, 51.25, 45.75, 75.77, 74.76, 65.85, 60.26, 46.72, 75.71, 70.38, 55.68, 47.57, 35.03, 75.86, 72.93, 52.61, 45.07, 42.66, 75.87, 67.35, 49.95, 44.17, 36.13, 75.86, 74.23, 73.14, 68.70, 65.53, 75.86, 73.71, 66.85, 65.88, 65.50, 75.77, 75.71, 73.76, 72.56, 70.81, 75.72, 71.60, 60.99, 56.74, 50.02, 75.77, 71.53, 57.84, 51.33, 60.51, 75.72, 70.38, 54.25, 51.12, 47.15, 47.47, 48.42, 66.22, 62.85, 50.08, 57.59, 74.86, 74.11, 72.91, 70.63, 68.95, 75.07, 74.78, 74.13, 73.69, 72.93, 75.61, 75.76, 75.61, 75.64, 75.74, 75.79, 75.42, 75.41, 75.44, 74.41, 71.74, 67.85, 67.28, 66.05, 64.80, 64.67, 60.04, 58.04, 58.36, 56.98, 61.85, 58.21, 53.90, 51.95, 49.15]
		base_MA_nvidia_H100k = [90.13, 86.45, 76.12, 62.28, 55.82, 52.77, 85.90, 76.37, 59.18, 41.88, 33.32, 90.20, 88.75, 79.93, 66.22, 64.35, 90.67, 74.70, 54.10, 46.45, 40.38, 88.33, 49.00, 45.50, 49.23, 56.60, 90.45, 84.02, 68.22, 59.77, 48.10, 89.60, 68.20, 53.90, 53.27, 56.80, 88.50, 42.42, 59.80, 64.80, 66.13, 84.13, 19.63, 15.78, 16.60, 27.92, 90.45, 82.80, 44.28, 38.90, 38.95, 90.18, 88.40, 70.87, 55.65, 38.12, 90.05, 87.65, 76.12, 72.00, 57.97, 89.53, 72.08, 51.45, 49.10, 43.67, 90.18, 82.93, 47.35, 33.15, 60.37, 88.90, 62.92, 41.98, 40.70, 41.00, 17.47, 3.85, 71.62, 30.85, 14.40, 24.08, 81.87, 75.13, 69.60, 62.98, 60.38, 84.78, 85.05, 83.27, 82.88, 80.20, 89.98, 89.30, 87.88, 84.97, 82.97, 85.38, 79.07, 78.52, 75.82, 71.33, 66.95, 50.77, 50.62, 41.73, 35.00, 43.73, 30.23, 25.65, 25.83, 23.43, 30.60, 22.12, 18.33, 17.72, 13.67]
		base_MA_nvidia_Ads = [91.08, 89.82, 88.59, 82.77, 68.88, 53.59, 89.50, 85.97, 69.65, 56.93, 47.25, 88.77, 49.30, 38.89, 4.93, 0.00, 91.53, 86.72, 76.05, 71.64, 63.54, 91.53, 85.27, 74.16, 69.70, 60.50, 79.51, 24.07, 6.00, 2.01, 0.33, 88.28, 39.15, 24.67, 20.61, 4.55, 81.14, 26.26, 6.26, 2.10, 0.23, 88.54, 39.43, 25.26, 21.15, 13.87, 87.98, 83.38, 64.17, 59.55, 33.71, 89.36, 52.85, 31.35, 49.30, 33.71, 91.81, 72.81, 49.14, 43.25, 30.95, 77.43, 23.76, 6.89, 6.44, 2.68, 67.83, 8.50, 0.37, 0.14, 0.00, 88.38, 34.34, 20.61, 11.18, 2.15, 15.92, 0.00, 28.99, 2.12, 37.07, 1.26, 86.04, 80.37, 71.08, 63.52, 57.77, 78.52, 74.39, 70.28, 67.81, 63.82, 89.73, 89.50, 90.15, 90.10, 89.96, 89.94, 89.26, 87.32, 81.61, 74.67, 18.53, 11.81, 11.83, 5.77, 2.57, 14.43, 10.85, 10.22, 11.06, 10.57, 12.89, 11.93, 9.85, 9.69, 8.43]
		base_MA_nvidia_Waymo = [48.34, 50.26, 51.00, 51.51, 52.27, 50.34, 48.51, 47.80, 42.63, 31.52, 21.89, 48.65, 49.73, 50.69, 51.45, 38.73, 49.32, 48.01, 47.48, 47.31, 45.06, 49.05, 36.97, 24.67, 23.91, 23.50, 49.12, 49.06, 50.00, 48.91, 51.26, 48.72, 34.90, 36.12, 40.04, 45.14, 48.90, 38.96, 51.52, 52.60, 53.20, 49.57, 45.67, 41.36, 40.43, 38.36, 47.79, 41.04, 45.21, 44.04, 39.82, 49.69, 49.17, 46.46, 41.48, 39.47, 48.93, 48.17, 43.69, 40.25, 29.30, 48.91, 47.44, 45.87, 45.97, 45.73, 49.16, 45.97, 31.03, 27.86, 31.85, 50.30, 49.85, 42.17, 36.61, 28.11, 44.30, 54.20, 50.33, 47.34, 54.00, 50.71, 48.97, 49.04, 48.32, 46.72, 47.00, 49.28, 49.27, 49.32, 49.51, 49.61, 49.79, 49.01, 49.85, 48.87, 50.41, 49.24, 49.30, 50.13, 49.21, 45.13, 42.08, 26.54, 26.44, 14.63, 9.13, 34.91, 26.00, 22.35, 24.18, 21.86, 25.69, 19.71, 16.34, 16.04, 14.07]
		base_MA_commaai_B = [81.96, 82.26, 82.50, 80.86, 77.56, 70.89, 82.71, 80.54, 77.74, 72.71, 67.68, 73.54, 49.49, 33.33, 54.32, 52.77, 81.76, 64.61, 51.73, 48.51, 46.67, 82.11, 57.95, 46.76, 45.57, 49.43, 82.41, 65.48, 51.93, 50.00, 47.38, 81.85, 66.04, 58.04, 57.41, 56.25, 82.65, 71.31, 60.51, 57.32, 50.95, 82.62, 70.89, 59.55, 58.21, 53.18, 81.43, 74.46, 66.19, 61.52, 53.84, 82.32, 67.80, 50.74, 53.04, 53.24, 83.15, 81.55, 73.15, 68.93, 57.14, 81.70, 74.11, 62.14, 58.27, 52.41, 82.86, 72.71, 47.02, 39.94, 37.68, 82.29, 76.25, 63.90, 60.45, 54.85, 54.64, 32.50, 36.88, 48.99, 60.65, 57.23, 82.23, 79.52, 78.07, 75.80, 73.45, 81.52, 80.18, 79.37, 78.01, 76.85, 82.35, 83.04, 82.62, 82.29, 83.01, 82.50, 82.11, 81.40, 80.15, 77.92, 66.99, 59.20, 57.92, 57.68, 57.23, 61.46, 60.51, 59.46, 60.18, 59.52, 62.02, 58.33, 55.74, 54.79, 52.38]
		base_MA_commaai_Hc = [74.23, 72.58, 73.46, 71.61, 67.57, 61.85, 73.53, 71.94, 70.63, 67.87, 63.47, 71.71, 69.08, 70.35, 67.50, 71.11, 74.08, 68.66, 57.09, 52.91, 47.09, 72.66, 51.53, 33.60, 31.92, 31.87, 73.76, 65.32, 46.35, 42.22, 39.09, 73.14, 66.23, 47.30, 42.39, 36.36, 73.41, 69.43, 58.54, 53.33, 45.09, 73.71, 56.33, 29.62, 20.13, 10.31, 73.16, 68.32, 65.73, 61.29, 50.52, 73.96, 70.28, 60.31, 58.26, 50.35, 72.66, 73.38, 71.96, 71.35, 67.02, 72.73, 71.06, 62.02, 58.28, 51.32, 72.86, 72.61, 63.44, 60.89, 65.07, 72.44, 66.53, 38.41, 26.26, 15.85, 69.68, 68.80, 69.91, 63.80, 36.70, 67.35, 72.96, 72.49, 70.90, 69.66, 68.61, 71.74, 70.00, 70.10, 69.65, 69.68, 72.39, 73.13, 73.31, 73.01, 72.93, 73.14, 71.99, 72.98, 72.64, 72.08, 64.44, 47.40, 47.97, 41.03, 30.69, 45.75, 32.88, 28.01, 28.84, 26.96, 53.53, 50.27, 46.15, 46.30, 43.89]
		base_MA_commaai_Ads = [79.34, 78.34, 78.08, 76.14, 67.81, 54.53, 78.76, 77.24, 72.99, 64.31, 54.95, 39.54, 5.72, 8.96, 3.06, 2.33, 77.33, 38.47, 4.20, 2.01, 1.31, 77.80, 49.56, 30.56, 29.74, 30.42, 78.27, 41.99, 4.55, 1.42, 0.63, 78.78, 62.98, 45.19, 40.64, 31.72, 75.00, 35.41, 22.53, 21.03, 17.79, 78.97, 32.66, 14.38, 9.92, 5.35, 78.66, 75.68, 59.59, 48.04, 44.89, 77.71, 70.17, 43.58, 32.96, 45.80, 78.06, 74.53, 63.94, 58.01, 46.17, 76.75, 40.45, 12.09, 9.64, 10.41, 73.83, 18.79, 1.73, 1.70, 1.82, 78.01, 43.77, 12.32, 4.53, 0.96, 11.41, 1.52, 6.82, 1.61, 33.59, 1.33, 77.66, 75.19, 71.36, 67.62, 61.74, 72.90, 70.28, 66.53, 64.17, 59.45, 78.22, 79.32, 78.36, 77.59, 78.38, 78.71, 77.92, 77.17, 76.28, 71.71, 25.75, 14.40, 14.85, 10.60, 6.68, 15.92, 11.95, 9.76, 10.74, 9.41, 21.66, 18.51, 18.09, 18.02, 18.65]
		
		base_MA_resnet_Hc = [82.20, 82.17, 81.52, 77.12, 71.86, 68.35, 82.15, 80.59, 75.46, 70.70, 67.80, 80.30, 76.37, 71.30, 66.95, 68.51, 82.05, 78.40, 70.93, 67.58, 62.02, 82.03, 77.71, 71.46, 70.46, 70.43, 82.12, 82.13, 82.03, 81.37, 80.24, 82.08, 80.92, 76.69, 74.79, 69.78, 82.10, 81.90, 78.45, 75.89, 72.19, 82.07, 77.19, 67.70, 62.50, 57.03, 82.03, 81.77, 81.55, 79.64, 77.56, 82.23, 81.65, 78.62, 77.97, 77.59, 82.10, 81.59, 80.10, 79.72, 77.91, 82.20, 80.29, 74.33, 72.18, 66.75, 82.10, 79.14, 71.28, 62.59, 58.34, 82.18, 75.54, 64.34, 64.10, 63.09, 70.26, 63.32, 69.95, 72.26, 65.15, 62.50, 82.07, 81.92, 80.77, 78.94, 77.66, 81.05, 80.54, 79.92, 79.32, 78.60, 82.17, 82.25, 82.17, 81.92, 81.83, 82.18, 82.10, 82.25, 81.95, 82.03, 74.04, 68.96, 68.33, 65.50, 64.14, 69.18, 64.60, 62.69, 62.84, 61.99, 64.97, 61.75, 57.91, 56.14, 52.91]

		if BN_flag == 5:
			if "trainB" in train_folder:
				base_MA = np.array(base_MA_commaai_B)
			elif "trainHc" in train_folder:
				base_MA = np.array(base_MA_commaai_Hc)
			elif "trainAds" in train_folder:
				base_MA = np.array(base_MA_commaai_Ads)
		elif BN_flag == 8:
			if "trainB" in train_folder:
				base_MA = np.array(base_MA_resnet_B)
			elif "trainHc" in train_folder:
				base_MA = np.array(base_MA_resnet_Hc)
			elif "trainAds" in train_folder:
				base_MA = np.array(base_MA_resnet_Ads)
		else:
			if "trainB" in train_folder:
				base_MA = np.array(base_MA_nvidia_B)
			elif "trainHc" in train_folder:
				base_MA = np.array(base_MA_nvidia_Hc)
			elif "trainHonda100k" in train_folder:
				base_MA = np.array(base_MA_nvidia_H100k)
			elif "trainAds" in train_folder:
				base_MA = np.array(base_MA_nvidia_Ads)
			elif "trainWaymo" in train_folder:
				base_MA = np.array(base_MA_nvidia_Waymo)


		base_scene1 = base_MA[0]
		base_scene2 = base_MA[1:76]
		base_scene3 = base_MA[76:82]
		base_scene4 = base_MA[82:117]


		for MA in MA_list:
			print("{:.2f}\t".format(MA))

		res_scene1 = MA_list[0]
		res_scene2 = MA_list[1:76]
		res_scene3 = MA_list[76:82]
		res_scene4 = MA_list[82:117]


		print("scene1\t{:.2f}".format(res_scene1))
		print("scene2\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene2), np.min(res_scene2), np.max(res_scene2)))
		print("scene3\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene3), np.min(res_scene3), np.max(res_scene3)))
		print("scene4\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene4), np.min(res_scene4), np.max(res_scene4)))

		print("scene1\t{:.2f}".format(res_scene1-base_scene1))
		print("scene2\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene2-base_scene2), np.min(res_scene2-base_scene2), np.max(res_scene2-base_scene2)))
		print("scene3\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene3-base_scene3), np.min(res_scene3-base_scene3), np.max(res_scene3-base_scene3)))
		print("scene4\taverage\t{:.2f}\tmin\t{:.2f}\tmax\t{:.2f}".format(np.mean(res_scene4-base_scene4), np.min(res_scene4-base_scene4), np.max(res_scene4-base_scene4)))



		MA_list = MA_list / 100.
		base_MA = base_MA / 100.
		corruption_accs_1, corruption_accs_2, corruption_accs_3, corruption_accs_4 = reformal_corruption_accs(MA_list)
		base_accs_1, base_accs_2, base_accs_3, base_accs_4 = reformal_corruption_accs(base_MA)

		# corruption_accs_1, corruption_accs_2, corruption_accs_3, corruption_accs_4 = reformal_corruption_accs(np.array(MA_list_commaai_Ads_augmix) / 100.)
		# base_accs_1, base_accs_2, base_accs_3, base_accs_4 = reformal_corruption_accs(np.array(base_MA_commaai_Ads) / 100.)

		mce1 = compute_mce(corruption_accs_1, base_accs_1)
		mce2 = compute_mce(corruption_accs_2, base_accs_2)
		mce3 = compute_mce(corruption_accs_3, base_accs_3)
		mce4 = compute_mce(corruption_accs_4, base_accs_4)

		print("mCE1", mce1)
		print("mCE2", mce2)
		print("mCE3", mce3)
		print("mCE4", mce4)



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='batch train test')
	parser.add_argument('--gpu_id', required=False, default=0, metavar="gpu_id", help='gpu id (0/1)')

    parser.add_argument('--train_folder', type=str, default='trainB', choices=['trainB','trainHonda100k','trainAds','trainHc','trainWaymo'])
    parser.add_argument('--val_folder', type=str, default='valB', choices=['valB','valHonda100k','valAds','valHc','valWaymo'])
    parser.add_argument('--with_FFT', default=False, choices=[True, False])
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--dataset_root', type=str, default='./Data/udacityA_nvidiaB/')
    parser.add_argument('--output_root', type=str, default='./Data/udacityA_nvidiaB_results/')
    parser.add_argument('--BN_flag', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='')

	args = parser.parse_args()

	if (args.gpu_id != None):
		os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
	print("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])


	DATASET_ROOT = args.dataset_root
	OUTPUT_ROOT = args.output_root
	TRAIN_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "/train_results/")
	TEST_OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, "/test_results/")

	if not os.path.exists(OUTPUT_ROOT):
		os.mkdir(OUTPUT_ROOT)

	if not os.path.exists(TRAIN_OUTPUT_ROOT):
		os.mkdir(TRAIN_OUTPUT_ROOT)

	if not os.path.exists(TEST_OUTPUT_ROOT):
		os.mkdir(TEST_OUTPUT_ROOT)


	multi_factor_search(args)