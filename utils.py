import cv2
import numpy as np
import tensorflow as tf
import scipy.io as scio


def gaussian_kernel_2d(kernel_size=3, sigma=0.):
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


def truncation_normal_distribution(standard_variance):
    return tf.truncated_normal_initializer(0.0, standard_variance)


def structural_similarity_index_metric(feature, labels, params=None):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    weight = gaussian_kernel_2d(11, 1.5)
    weight = tf.constant(weight)
    weight = tf.reshape(weight, [11, 11, 1, 1])
    weight = tf.cast(weight, tf.float32)
    mean_f = tf.nn.conv2d(feature, weight, [1, 1, 1, 1], padding="SAME")
    mean_y = tf.nn.conv2d(labels, weight, [1, 1, 1, 1], padding="SAME")
    mean_f_mean_y = tf.multiply(mean_f, mean_y)
    square_mean_f = tf.multiply(mean_f, mean_f)
    square_mean_y = tf.multiply(mean_y, mean_y)
    variance_f = tf.nn.conv2d(tf.multiply(feature, feature), weight, [1, 1, 1, 1], padding="SAME") - square_mean_f
    variance_y = tf.nn.conv2d(tf.multiply(labels, labels), weight, [1, 1, 1, 1], padding="SAME") - square_mean_y
    variance_fy = tf.nn.conv2d(tf.multiply(feature, labels), weight, [1, 1, 1, 1], padding="SAME") - mean_f_mean_y
    ssim = ((2*mean_f_mean_y + c1)*(2*variance_fy + c2)) / \
           ((square_mean_f + square_mean_y + c1)*(variance_f + variance_y + c2))
    return 1 - tf.reduce_mean(ssim, reduction_indices=[1, 2, 3])
