import numpy as np
import scipy.io as scio
import cv2
from PIL import Image
import random

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


def image_augmentation(
        image_dir_path,
        ground_truth_dir_path,
        output_image_path,
        output_gt_path,
        num,
        crop_num=4):
    data_image = []
    data_groud_truth = []

    for i in range(num):
        img_path = image_dir_path + "/IMG_" + str(i + 1) + ".jpg"
        gt_path = ground_truth_dir_path + "/GT_IMG_" + str(i + 1) + ".mat"
        img = Image.open(img_path)
        height = img.size[1]
        weight = img.size[0]
        gt = generate_density_map(15, gt_path, height, weight)
        gt = np.reshape(gt, [height, weight, 1])
        if img.mode == 'RGB':
            img = np.reshape(img, [height, weight, 3])
        else:
            img = np.reshape(img, [height, weight, 1])
            img = np.tile(img, [1, 1, 3])

        for j in range(crop_num):
            left_y = random.randint(0, height // 2)
            left_x = random.randint(0, weight // 2)
            image = img[left_y:left_y + height // 2, left_x:left_x + weight // 2, :]
            ground_truth = gt[left_y:left_y + height // 2, left_x:left_x + weight // 2, :]
            mirror_img = image[:, ::-1, :]
            mirror_gt = ground_truth[:, ::-1, :]
            print(image.shape)
            print(mirror_img.shape)
            print(ground_truth.shape)
            print(mirror_gt.shape)
            data_image.append(image)
            data_groud_truth.append(ground_truth)
            data_image.append(mirror_img)
            data_groud_truth.append(mirror_gt)
            print("complete!")

    data_groud_truth = np.array(data_groud_truth)
    data_image = np.array(data_image)
    np.save(output_image_path, data_image)
    np.save(output_gt_path, data_groud_truth)


if __name__ == "__main__":
    image_dir_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/images"
    ground_truth_dir_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/ground_truth"
    output_image_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/images_train.npy"
    output_gt_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/gt_train.npy"
    image_augmentation(image_dir_path, ground_truth_dir_path, output_image_path, output_gt_path, 380, crop_num=4)
    print("complete!")
