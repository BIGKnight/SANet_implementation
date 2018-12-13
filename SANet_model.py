import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
import scipy.io as scio
import cv2
import tensorflow.contrib.slim as slim
import utils
result_output = open("/home/zzn/SANet_implementation-master/result_B_12.13.txt", "w")
image_train_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/images.npy"
gt_train_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/gt.npy"
image_eval_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/test_data/images.npy"
gt_eval_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/test_data/gt.npy"
batch_size = 1
epoch = 300
loss_c_weight = 0.001


def inception_arg_scope(weight_decay=4e-4, std=0.1, batch_norm_var_collection="moving_vars"):
    instance_norm_params = {
        # "decay": 0.9997,
        "epsilon": 0.001,
        "activation_fn": tf.nn.relu,
        "trainable": True,
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": [batch_norm_var_collection],
            "moving_variance": [batch_norm_var_collection]},
        "outputs_collections": {

        }
    }
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.truncated_normal_initializer(stddev=std),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.instance_norm,
                        normalizer_params=instance_norm_params) as sc:
        return sc

# variables_collections, output_collections, param_initializers


def encoder_unit(data_input, channel_output, layer_number):
    with tf.variable_scope(name_or_scope="branch_1x1_" + str(layer_number)):
        branch_1x1 = slim.conv2d(data_input, channel_output, [2, 2], 1, "same")
    with tf.variable_scope(name_or_scope="branch_3x3_" + str(layer_number)):
        branch_3x3_part_1 = slim.conv2d(data_input, 2*channel_output, [1, 1], 1, "SAME", scope="convolution_layer_1a")
        branch_3x3_part_2 = slim. \
            conv2d(branch_3x3_part_1, channel_output, [3, 3], 1, "SAME", scope="convolution_layer_1b")
    with tf.variable_scope(name_or_scope="branch_5x5_" + str(layer_number)):
        branch_5x5_part_1 = slim.conv2d(data_input, 2*channel_output, [1, 1], 1, "SAME", scope="convolution_layer_1a")
        branch_5x5_part_2 = slim. \
            conv2d(branch_5x5_part_1, channel_output, [5, 5], 1, "SAME", scope="convolution_layer_1c")
    with tf.variable_scope(name_or_scope="branch_7x7_" + str(layer_number)):
        branch_7x7_part_1 = slim.conv2d(data_input, 2*channel_output, [1, 1], 1, "SAME", scope="convolution_layer_1a")
        branch_7x7_part_2 = slim. \
            conv2d(branch_7x7_part_1, channel_output, [7, 7], 1, "SAME", scope="convolution_layer_1d")
    output = tf.concat([branch_1x1, branch_3x3_part_2, branch_5x5_part_2, branch_7x7_part_2], 3)
    return output


def encoder_head(data_input, channel_output):
    with tf.variable_scope(name_or_scope="branch_1x1_head"):
        branch_1x1 = slim.conv2d(data_input, channel_output, [2, 2], 1, "same")
    with tf.variable_scope(name_or_scope="branch_3x3_head"):
        branch_3x3 = slim. \
            conv2d(data_input, channel_output, [3, 3], 1, "SAME", scope="convolution_layer_1b")
    with tf.variable_scope(name_or_scope="branch_5x5_head"):
        branch_5x5 = slim. \
            conv2d(data_input, channel_output, [5, 5], 1, "SAME", scope="convolution_layer_1c")
    with tf.variable_scope(name_or_scope="branch_7x7_head"):
        branch_7x7 = slim. \
            conv2d(data_input, channel_output, [7, 7], 1, "SAME", scope="convolution_layer_1d")
    output = tf.concat([branch_1x1, branch_3x3, branch_5x5, branch_7x7], 3)
    return output


def scale_aggregation_network(features):
    with slim.arg_scope(inception_arg_scope()):
        features = tf.cast(features, tf.float32)
        feature_map_encoder = encoder_head(features, 16)
        feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_2x2")
        feature_map_encoder = encoder_unit(feature_map_encoder, 32, 1)
        feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_2x2")
        feature_map_encoder = encoder_unit(feature_map_encoder, 32, 2)
        feature_map_encoder = slim.max_pool2d(feature_map_encoder, [2, 2], 2, "SAME", scope="max_pooling_2x2")
        feature_map_encoder = encoder_unit(feature_map_encoder, 32, 3)

        density_map_estimator = slim.conv2d(feature_map_encoder, 64, [9, 9], 1, "SAME")
        density_map_estimator = slim.conv2d_transpose(density_map_estimator, 64, [2, 2], stride=2)
        density_map_estimator = slim.conv2d(density_map_estimator, 32, [7, 7], 1, "SAME")
        density_map_estimator = slim.conv2d_transpose(density_map_estimator, 32, [2, 2], stride=2)
        density_map_estimator = slim.conv2d(density_map_estimator, 16, [5, 5], 1, "SAME")
        density_map_estimator = slim.conv2d_transpose(density_map_estimator, 16, [2, 2], stride=2)
        density_map_estimator = slim.conv2d(density_map_estimator, 16, [3, 3], 1, "SAME")
    density_map_estimator = slim.conv2d(density_map_estimator, 1, [1, 1], 1, "SAME", normalizer_fn=None, normalizer_params=None)
# NHWC
    return density_map_estimator


if __name__ == "__main__":
    image_train = np.load(image_train_path)
    gt_train = np.load(gt_train_path)
    image_eval = np.load(image_eval_path)
    gt_eval = np.load(gt_eval_path)

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
    estimated_density_map = scale_aggregation_network(x)
    estimated_counting = tf.reduce_sum(estimated_density_map, reduction_indices=[1, 2, 3], name="crowd_counting")
    ground_truth_counting = tf.reduce_sum(y, reduction_indices=[1, 2, 3])
    ground_truth_counting = tf.cast(ground_truth_counting, tf.float32)

    loss_e = tf.losses.mean_squared_error(y, predictions=estimated_density_map)
    loss_c = structural_similarity_index_metric(estimated_density_map, y)
    loss = tf.add(loss_e, tf.multiply(loss_c_weight, loss_c))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'MAE': tf.metrics.mean(tf.abs(tf.subtract(estimated_counting, ground_truth_counting)), name="MAE"),
        'MSE': tf.metrics.root_mean_squared_error(ground_truth_counting, predictions=estimated_counting, name="MSE")
    }
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        image_train_num = len(image_train)
        step = 0
        for i in range(epoch):
            for j in range(image_train_num // batch_size):
                start = (j*batch_size) % image_train_num
                end = min(start + batch_size, image_train_num)
                sess.run(train_op, feed_dict={x: image_train[start:end], y: gt_train[start:end]})
                step = step + 1
                if step % 50 == 0:
                    loss, eval_metric_ops = sess.run([loss, eval_metric_ops], feed_dict={x: image_eval, y: gt_eval})
                    print(loss)
                    print(eval_metric_ops)
