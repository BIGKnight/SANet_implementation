import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
import scipy.io as scio
import cv2
import tensorflow.contrib.slim as slim
from PIL import Image

# 制作二进制FTRecord文件


def gaussian_kernel_2d(kernel_size, sigma):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


def generate_density_map(gaussian_radius, gt_data, N, M):
    R = gaussian_radius
    gaussian_kernel = gaussian_kernel_2d(R, 4)
    gt = scio.loadmat(gt_data)['image_info']
    gt = gt[0][0][0][0]
    coordinate = gt[0]
    density_map = np.zeros([N, M], dtype=float)
    for y, x in coordinate:
        x = round(x - 1, 0)
        y = round(y - 1, 0)
        for i in range(int(x - R / 2), int(x + R / 2 + 1)):
            if i < 0 or i > (N - 1):
                continue
            for j in range(int(y - R / 2), int(y + R / 2 + 1)):
                if j < 0 or j > (M - 1):
                    continue
                else:
                    density_map[i][j] = density_map[i][j] + \
                                        gaussian_kernel[i - int(x - R / 2) - 1][j - int(y - R / 2) - 1]
    return density_map


image_dir_path = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/part_C_final/train_data/images"
ground_truth_dir_path = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/" \
                        "part_C_final/train_data/ground_truth"
output_path = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/" \
                                         "part_C_final/train_data/crowd_train_data.tfrecords"


def make_tfrecord(data_dir_path, gt_dir_path, output_path, example_num):
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in range(example_num):
        img_path = data_dir_path + "/IMG_" + str(i + 1) + ".jpg"
        gt_path = gt_dir_path + "/GT_IMG_" + str(i + 1) + ".mat"
        img = Image.open(img_path)
        height = img.size[1]
        weight = img.size[0]
        channel = 1 if img.mode == 'L' else 3
        density_map_gt = generate_density_map(15, gt_path, height, weight)
        density_map_gt = np.reshape(density_map_gt, [height, weight, 1])
        img_raw = img.tobytes()  # 将图片转化为二进制的格式
        density_map_gt = density_map_gt.tobytes()
        # example对象对label和image数据进行封装
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "density_map_gt": tf.train.Feature(bytes_list=tf.train.BytesList(value=[density_map_gt])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "HWC": tf.train.Feature(int64_list=tf.train.Int64List(value=[height, weight, channel]))
                }
            )
        )
        # 将序列转为字符串
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    make_tfrecord(image_dir_path, ground_truth_dir_path, output_path, 5)

