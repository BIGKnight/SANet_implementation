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
    return result


if __name__ == "__main__":
    R = 15
    gt_data = data_traversal(gt_file, '.mat')
    image_data = data_traversal(image_file, '.jpg')
    gaussian_kernel = gaussian_kernel_2d(R, 4)
    for k in range(len(image_data)):
        gt = scio.loadmat(gt_file + "/GT_IMG_" + str(k + 1) + ".mat")['image_info']
        gt = gt[0][0][0][0]
        count = gt[1][0][0]
        coordinate = gt[0]
        image = cv2.imread(image_file + "/IMG_" + str(k + 1) + ".jpg")
        N = image.shape[0]
        M = image.shape[1]
        density_map = np.zeros([N, M], dtype=float)
        for y, x in coordinate:
            x = round(x - 1, 0)
            y = round(y - 1, 0)
            flag = 0
            for i in range(int(x - R / 2), int(x + R / 2 + 1)):
                if i < 0 or i > (N - 1):
                    continue
                for j in range(int(y - R / 2), int(y + R / 2 + 1)):
                    if j < 0 or j > (M - 1):
                        continue
                    else:
                        density_map[i][j] = density_map[i][j] + gaussian_kernel[i - int(x - R / 2) - 1][j - int(y - R / 2) - 1]

        # max_den = density_map.max()
        # den_map = np.zeros([N, M, 3], dtype=np.float)
        # for X in range(N):
        #     for Y in range(M):
        #         pixel = 255 * density_map[X][Y] / max_den
        #
        #         den_map[X][Y] = mp[int(pixel)] * 255
        #         den_map[X][Y] = [int(ele) for ele in den_map[X][Y]]
        # cv2.imwrite("/Users/elrond/Desktop/" + 'density_map_' + str(k + 1) + '.jpg', den_map)