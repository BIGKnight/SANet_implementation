import tensorflow as tf
import tensorflow.contrib.slim as slim


def inception_arg_scope(weight_decay=4e-4, std=3, batch_norm_var_collection="moving_vars"):
    instance_norm_params = {
        # "decay": 0.9997,
        "epsilon": 1e-6,
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
                        weights_initializer=tf.glorot_uniform_initializer(),
                        activation_fn=tf.nn.relu) as sc:
        return sc
#  normalizer_fn=slim.instance_norm,
#                         normalizer_params=instance_norm_params
# variables_collections, output_collections, param_initializers


def encoder_unit(data_input, channel_output, layer_number):
    with tf.variable_scope(name_or_scope="branch_1x1_" + str(layer_number)):
        branch_1x1 = slim.conv2d(data_input, channel_output, [1, 1], 1, "same")
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
    slim.contrib_framework.get_variables_to_restore()

    return output


def encoder_head(data_input, channel_output):
    with tf.variable_scope(name_or_scope="branch_1x1_head"):
        branch_1x1 = slim.conv2d(data_input, channel_output, [1, 1], 1, "same")
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
        # input instance normalization
        features = tf.divide(features, 255)
        features = slim.instance_norm(features, epsilon=1e-6)
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