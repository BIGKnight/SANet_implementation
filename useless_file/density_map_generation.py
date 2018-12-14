import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
from PIL import Image
import scipy.io as scio
import tensorflow.contrib.slim as slim
import cv2

gt_file = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/part_C_final/train_data/ground_truth"
color_map = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/map.mat"
image_file = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/part_C_final/train_data/images"
R = 15
mp = scio.loadmat(color_map)
mp = mp['c']
mp = mp[::-1]


def gaussian_kernel_2d(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


def data_traversal(file_dir, format):
    result = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == format:
                result.append(os.path.join(root, file))
    return resul