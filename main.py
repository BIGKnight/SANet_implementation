import tensorflow as tf
import numpy as np
import utils
import SANet_model
result_output = open("/home/zzn/SANet_implementation-master/result_B_12.13.txt", "w")
image_train_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/images.npy"
gt_train_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/train_data/gt.npy"
image_eval_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/test_data/images.npy"
gt_eval_path = "/media/zzn/922E52FA2E52D737/SANet/part_B_final/test_data/gt.npy"
batch_size = 1
epoch = 300
loss_c_weight = 0.001


if __name__ == "__main__":
    image_train = np.load(image_train_path)
    gt_train = np.load(gt_train_path)
    image_eval = np.load(image_eval_path)
    gt_eval = np.load(gt_eval_path)

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
    estimated_density_map = SANet_model.scale_aggregation_network(x)
    estimated_counting = tf.reduce_sum(estimated_density_map, reduction_indices=[1, 2, 3], name="crowd_counting")
    ground_truth_counting = tf.reduce_sum(y, reduction_indices=[1, 2, 3])
    ground_truth_counting = tf.cast(ground_truth_counting, tf.float32)

    loss_e = tf.losses.mean_squared_error(y, predictions=estimated_density_map)
    loss_c = utils.structural_similarity_index_metric(estimated_density_map, y)
    loss = tf.add(loss_e, tf.multiply(loss_c_weight, loss_c))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss,
                                                                               global_step=tf.train.get_global_step())
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
                start = (j * batch_size) % image_train_num
                end = min(start + batch_size, image_train_num)
                sess.run(train_op, feed_dict={x: image_train[start:end], y: gt_train[start:end]})
                step = step + 1
                if step % 50 == 0:
                    loss, eval_metric_ops = sess.run([loss, eval_metric_ops], feed_dict={x: image_eval, y: gt_eval})
                    print(loss)
                    print(eval_metric_ops)