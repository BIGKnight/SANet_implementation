import numpy as np
import scipy.io as scio
import scipy
import cv2
from PIL import Image
import random
from utils import get_density_map_gaussian


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
        points = scio.loadmat(gt_path)['image_info'][0][0][0][0][0]
        gt = get_density_map_gaussian(height, weight, points, False, 5)
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
    image_dir_path = "/home/zzn/part_B_final/train_data/images"
    ground_truth_dir_path = "/home/zzn/part_B_final/train_data/ground_truth"
    output_image_path = "/home/zzn/part_B_final/train_data/images_train.npy"
    output_gt_path = "/home/zzn/part_B_final/train_data/gt_train.npy"
    image_augmentation(image_dir_path, ground_truth_dir_path, output_image_path, output_gt_path, 380, crop_num=4)
    print("complete!")
