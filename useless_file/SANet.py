import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
import scipy.io as scio
import cv2
import tensorflow.contrib.slim as slim


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


def scale_aggregation_network(features, labels, mode, params):
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
        density_map_estimator = slim.\
            conv2d(density_map_estimator, 1, [1, 1], 1, "SAME", normalizer_fn=None, normalizer_params=None)
# NHWC
        crowd_counting_estimated = tf.reduce_sum(density_map_estimator, reduction_indices=[1, 2, 3], name="crowd_counting")
        ground_truth_counting = tf.reduce_sum(labels, reduction_indices=[1, 2, 3])
        density_map = density_map_estimator
        predictions = {
            'crowd_counting': crowd_counting_estimated,
            'density_map': density_map
        }

        density_map = tf.cast(density_map, tf.float32)
        labels = tf.cast(labels, tf.float32)
        ground_truth_counting = tf.cast(ground_truth_counting, tf.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss_e = tf.losses.mean_squared_error(labels, predictions=density_map)
        weight = 0.001
        loss_c = structural_similarity_index_metric(density_map_estimator, labels)
        loss = tf.add(loss_e, tf.multiply(weight, loss_c))

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).\
                minimize(loss=loss, global_step=tf.train.get_global_step())
            # train_op = tf.train.AdamOptimizer(learning_rate=1e-6).\
            #     minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            'MAE': tf.metrics.mean(tf.abs(tf.subtract(crowd_counting_estimated, ground_truth_counting)), name="MAE"),
            'MSE': tf.metrics.root_mean_squared_error(ground_truth_counting, predictions=crowd_counting_estimated, name="MSE")
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    file = open("/home/zzn/SANet_implementation-master/result_A_12.13.txt", "w")
    with tf.Session() as sess:
        image_train = np.load("/media/zzn/922E52FA2E52D737/SANet/part_A_final/train_data/images.npy")
        gt_train = np.load("/media/zzn/922E52FA2E52D737/SANet/part_A_final/train_data/gt.npy")
        image_eval = np.load("/media/zzn/922E52FA2E52D737/SANet/part_A_final/test_data/images.npy")
        gt_eval = np.load("/media/zzn/922E52FA2E52D737/SANet/part_A_final/test_data/gt.npy")
        estimator = tf.estimator.Estimator(model_fn=scale_aggregation_network,
                                           model_dir="/media/zzn/922E52FA2E52D737/SANet/"
                                                     "part_A_final/scale_aggregation_model")
        for i in range(500):
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=image_train,
                y=gt_train,
                batch_size=1,
                num_epochs=1,
                shuffle=True
            )
            estimator.train(input_fn=train_input_fn)
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=image_eval,
                y=gt_eval,
                batch_size=1,
                num_epochs=1,
                shuffle=False
            )
            eval_result = estimator.evaluate(input_fn=eval_input_fn)
            file.write(str(eval_result))
            file.write("\r\n")
            print(eval_result)
    file.close()
    print("complete")


