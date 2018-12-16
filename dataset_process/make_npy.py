import numpy as np
import scipy.io as scio
from PIL import Image
from utils import get_density_map_gaussian


def make_npy(image_dir_path, ground_truth_dir_path, output_image_path, output_gt_path, num):
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

        data_image.append(img)
        data_groud_truth.append(gt)

    data_groud_truth = np.array(data_groud_truth)
    data_image = np.array(data_image)
    np.save(output_image_path, data_image)
    np.save(output_gt_path, data_groud_truth)


if __name__ == "__main__":
    image_dir_path = "/home/zzn/part_B_final/test_data/images"
    ground_truth_dir_path = "/home/zzn/part_B_final/test_data/ground_truth"
    output_image_path = "/home/zzn/part_B_final/test_data/images_test.npy"
    output_gt_path = "/home/zzn/part_B_final/test_data/gt_test.npy"
    make_npy(image_dir_path, ground_truth_dir_path, output_image_path, output_gt_path, 316)
    print("complete!")