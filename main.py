# %load /home/zzn/PycharmProjects/SANet_implementation/main.py
import tensorflow as tf
import numpy as np
import utils
import sys
import matplotlib.pyplot as plt
import SANet_model

result_output = open("/home/zzn/SANet_implementation-master/result_B_12.13.txt", "w")
image_train_path = "/home/zzn/part_B_final/train_data/images_train.npy"
gt_train_path = "/home/zzn/part_B_final/train_data/gt_train.npy"
image_validate_path = "/home/zzn/part_B_final/train_data/images_validate.npy"
gt_validate_path = "/home/zzn/part_B_final/train_data/gt_validate.npy"
batch_size = 1
epoch = 300
loss_c_weight = 0.001


if __name__ == "__main__":
    image_train = np.load(image_train_path)
    gt_train = np.load(gt_train_path)
    image_validate = np.load(image_validate_path)
    gt_validate = np.load(gt_validate_path)

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")
    estimated_density_map = SANet_model.scale_aggregation_network(x)
    estimated_counting = tf.reduce_sum(estimated_density_map, reduction_indices=[1, 2, 3], name="crowd_counting")
    ground_truth_counting = tf.reduce_sum(y, reduction_indices=[1, 2, 3])
    ground_truth_counting = tf.cast(ground_truth_counting, tf.float32)

    loss_e = tf.losses.mean_squared_error(y, predictions=estimated_density_map)
    loss_c = utils.structural_similarity_index_metric(estimated_density_map, y)
    loss = tf.add(loss_e, tf.multiply(loss_c_weight, loss_c))

    train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'MAE': tf.reduce_mean(tf.abs(tf.subtract(estimated_counting, ground_truth_counting)), axis=0, name="MAE"),
        'MSE': tf.reduce_mean(tf.square(tf.subtract(ground_truth_counting, estimated_counting)), axis=0, name="MSE"),
    }

    MAE = 19970305
    MSE = 19970305
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        image_train_num = len(image_train)
        step = 0
        for i in range(epoch):
            for j in range(image_train_num // batch_size):

                # train
                start = (j * batch_size) % image_train_num
                end = min(start + batch_size, image_train_num)
                sess.run(train_op, feed_dict={x: image_train[start:end], y: gt_train[start:end]})
                step = step + 1

                # validate
                if step % 50 == 0:
                    loss_ = []
                    MAE_ = []
                    MSE_ = []
                    for k in range(len(image_validate // batch_size)):
                        loss_eval, metric_eval = sess.run([loss, eval_metric_ops],
                                                          feed_dict={x: image_validate[k:k + batch_size],
                                                                     y: gt_validate[k:k + batch_size]})
                        loss_.append(loss_eval)
                        MAE_.append(metric_eval['MAE'])
                        MSE_.append(metric_eval['MSE'])
                    loss_ = np.mean(loss_)
                    MAE_ = np.mean(MAE_)
                    RMSE = np.sqrt(np.mean(MSE_))
                    sys.stdout.write(
                        'In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\r'.format(step, i + 1, loss_, MAE_,
                                                                                          RMSE))
                    result_output.write(str(loss_) + "      " + "MAE: " + str(MAE_) + " MSE: " + str(RMSE) + "\r\n")
                    sys.stdout.flush()

                    fogure, (origin, density_gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
                    origin.imshow(image_validate[0])
                    origin.set_title('Origin Image')
                    density_gt.imshow(np.squeeze(gt_validate[0]), cmap=plt.cm.jet)
                    density_gt.set_title('ground_truth')
                    predict_den = np.squeeze(
                        sess.run(
                            estimated_density_map,
                            feed_dict={x: image_validate[0:1]})
                    )
                    pred.imshow(predict_den, cmap=plt.cm.jet)
                    plt.suptitle("one sample from the validate")
                    plt.show()
                    # save model
                    if MAE > MAE_:
                        MAE = MAE_
                        saver.save(sess, './checkpoint_dir/MyModel')
result_output.close()