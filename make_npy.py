import numpy as np
import scipy.io as scio
import cv2
from PIL import Image

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
data_image = []
data_groud_truth = []

for i in range(10):
    img_path = image_dir_path + "/IMG_" + str(i + 1) + ".jpg"
    gt_path = ground_truth_dir_path + "/GT_IMG_" + str(i + 1) + ".mat"
    img = Image.open(img_path)
    height = img.size[1]
    weight = img.size[0]
    channel = 1 if img.mode == 'L' else 3
    gt = generate_density_map(15, gt_path, height, weight)
    gt = np.reshape(gt, [height, weight, 1])
    if img.mode == 'RGB':
        img = np.reshape(img, [height, weight, 3])
        print(img.shape)
    else:
        img = np.reshape(img, [height, weight, 1])
        img = np.tile(img, [1, 1, 3])
        print(img.shape)

    data_image.append(img)
    data_groud_truth.append(gt)

data_groud_truth = np.array(data_groud_truth)
data_image = np.array(data_image)
np.save("/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/part_C_final/train_data/images.npy", data_image)
np.save("/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/part_C_final/train_data/gt.npy", data_groud_truth)