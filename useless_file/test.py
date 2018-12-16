import numpy as np
# import tensorflow as tf
# a = np.array([[[1., 2., 3.], [2., 1., 3.], [3., 1., 2.]], [[5., 2., 3.], [2., 1., 3.], [4., 1., 2.]]])
# b = np.array([[[1., 2., 3.], [2., 1., 3.], [3., 1., 2.]], [[5., 2., 3.], [2., 1., 3.], [4., 1., 2.]]])
# c = tf.constant([[[1., 2., 3.], [2., 1., 3.], [3., 1., 2.]], [[5., 2., 3.], [2., 1., 3.], [4., 1., 2.]]])
# d = tf.constant([1., 2.])
# # c = tf.divide(c, 5)
# f = tf.subtract(c[0], d[0])
# with tf.Session() as sess:
#     print(sess.run(f))

# # 迭代的计数器
# global_step = tf.Variable(0, trainable=False)
# # 迭代的+1操作
# increment_op = tf.assign_add(global_step, tf.constant(1))
# # 实例应用中，+1操作往往在`tf.train.Optimizer.apply_gradients`内部完成。
#
# # 创建一个根据计数器衰减的Tensor
# lr = tf.train.exponential_decay(0.1, global_step, decay_steps=1, decay_rate=0.9, staircase=False)
#
# # 把Tensor添加到观测中
# tf.scalar_summary('learning_rate', lr)
#
# # 并获取所有监测的操作`sum_opts`
# sum_ops = tf.merge_all_summaries()
#
# # 初始化sess
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)  # 在这里global_step被赋初值
#
# # 指定监测结果输出目录
# summary_writer = tf.train.SummaryWriter('/tmp/log/', sess.graph)
#
# # 启动迭代
# for step in range(0, 10):
#     s_val = sess.run(sum_ops)    # 获取serialized监测结果：bytes类型的字符串
#     summary_writer.add_summary(s_val, global_step=step)   # 写入文件
#     sess.run(increment_op)     # 计数器+1