import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
import scipy.io as scio
import cv2
import tensorflow.contrib.slim as slim
from PIL import Image

tfrecord_path = "/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/" \
         "part_C_final/train_data/crowd_train_data.tfrecords"


def read_tf_record(tf_record_path, example_num):
    file_name_queue = tf.train.string_input_producer([tf_record_path])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            "density_map_gt": tf.FixedLenFeature([], dtype=tf.string),
            'img_raw': tf.FixedLenFeature([], dtype=tf.string),
            "HWC": tf.FixedLenFeature([3], dtype=tf.int64)
        }
    )
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    HWC = tf.cast(features["HWC"], tf.int32)
    ground_truth = tf.decode_raw(features['density_map_gt'], np.float64)
    image = tf.cond(tf.equal(HWC[2], 1),
                    lambda: tf.reshape(image, [HWC[0], HWC[1]]),
                    lambda: tf.reshape(image, [HWC[0], HWC[1], HWC[2]])
                    )
    ground_truth = tf.reshape(ground_truth, [HWC[0], HWC[1], 1])
    mode = tf.cond(tf.equal(HWC[2], 1), lambda: tf.Variable("L"), lambda: tf.Variable("RGB"))
    result = []
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(example_num):
            train_image, gt, m = sess.run([image, ground_truth, mode])  # 在会话中取出image和label
            m = m.decode()
            img = Image.fromarray(train_image, mode=m)
            img.save("/Users/elrond/Documents/ShanghaiTech_Crowd_Counting_Dataset/part_C_final/train_data/" +
                     'IMG_' + str(i + 1) + '.jpg')  # 储存图片
            # print(train_image, gt)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    read_tf_record(tfrecord_path, 5)