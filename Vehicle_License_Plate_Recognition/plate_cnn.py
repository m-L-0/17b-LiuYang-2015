import numpy as np
import tensorflow as tf
import os
import random


# 读取TFRecord文件
def read_and_decode(tfrecords_file, batch_size=25):
    filename_queue = tf.train.string_input_producer([tfrecords_file])  #入队

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [48, 24])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=11580,
        min_after_dequeue=1000)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()  # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                image, label = sess.run([image_batch, label_batch])
                i += 1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
    coord.join(threads)
    b = np.empty((label.shape[0], 36))
    for i in range(label.shape[0]):
        for j in range(36):
            if j == label[i]:
                b[i][j] = 1.
            else:
                b[i][j] = 0.

    s, d, f = image.shape
    images = np.empty((s, d * f))
    for q in range(s):
        c = image[q]
        images[q] = c.reshape(d * f)
    return images, b

# 随机生成样本
def rand(image, label, batchsize):
    images = np.zeros((batchsize, 1152))
    labels = np.zeros((batchsize, 36))
    for i in range(batchsize):
        a = random.randint(0, image.shape[0] - 1)
        images[i] = image[a]
        labels[i] = label[a]
    return images, labels


# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir', 'cnn', '目录')
tf.app.flags.DEFINE_string('plate_cnn', 'plate_cnn', '模型名称:plate_cnn')
[X_train, y_train] = read_and_decode('train.tfrecords', 11580)
[X_val, y_val] = read_and_decode('Validation.tfrecords', 2895)

# 模型文件所在的文件夹，是否存在，如果不存在，则创建文件夹
ckpt = tf.train.latest_checkpoint(FLAGS.dir)
if not ckpt:
    if not os.path.exists(FLAGS.dir):
        os.mkdir(FLAGS.dir)
X_ = tf.placeholder(tf.float32, [None, 1152])
y_ = tf.placeholder(tf.float32, [None, 36])

# 把X转为卷积所需要的形式
X = tf.reshape(X_, [-1, 48, 24, 1])
# 第一层卷积：3×3×1卷积核32个 [3，3，1，32],h_conv1.shape=[-1, 48, 24, 32],学习32种特征
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

# 第一个pooling 层[-1, 48, 24, 32]->[-1, 24, 12, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [3，3，32，64],h_conv2.shape=[-1, 24, 12, 64]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个pooling 层,[-1, 24, 12, 64]->[-1, 14, 6, 64]
h_pool2 = max_pool_2x2(h_conv2)

#
W_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
h_pool2_flat = tf.reshape(h_pool3, [-1, 6 * 3 * 96])

# fc1
W_fc1 = weight_variable([6 * 3 * 96, 512])
b_fc1 = bias_variable([512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([512, 36])
b_fc2 = bias_variable([36])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 1.损失函数：cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 2.优化函数：AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 3.预测准确结果统计
# 预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
z = tf.argmax(y_conv, 1)
q = tf.argmax(y_, 1)
correct_prediction = tf.equal(z, q)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_acc_sum = tf.Variable(0.0)
batch_acc = tf.placeholder(tf.float32)
new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
update = tf.assign(test_acc_sum, new_test_acc_sum)
saver = tf.train.Saver(max_to_keep=2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(FLAGS.dir, sess.graph)

    ckpt = tf.train.latest_checkpoint(FLAGS.dir)
    step = 0
    if ckpt:
        saver.restore(sess=sess, save_path=ckpt)
        step = int(ckpt[len(os.path.join(FLAGS.dir, FLAGS.plate_cnn)) + 1:])

        check_point_path = '/home/miles/Vehicle_License_Plate_Recognition/cnn'  # 保存好模型的文件路径
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

# 训练
    for i in range(1):
        X_batch, y_batch = rand(X_train, y_train, 50)
        X, y = rand(X_val, y_val, 50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={X_: X,
                           y_: y,
                           keep_prob: 1.0})
            print("%d, Accuracy:%g" % (i, train_accuracy))
            ckptname = os.path.join(FLAGS.dir, FLAGS.plate_cnn)
            saver.save(sess, ckptname, global_step=i)
        train_step.run(feed_dict={X_: X_batch, y_: y_batch, keep_prob: 0.5})

    Y = np.zeros(2895)
    X_batch, y_batch = rand(X_val, y_val, 2895)
    Ytemp = y_conv.eval(feed_dict={X_: X_batch, keep_prob: 1.0})
    for i in range(2895):
        Y[i] = np.argmax(Ytemp[i])
    print("验证集正确率为 %g" % accuracy.eval(
        feed_dict={X_: X_batch,
                   y_: y_batch,
                   keep_prob: 1.0}))

label_str = '98AXP67H1GB0UREKCSVJNQYM32WLT4Z5DF'
co = 0
for j in range(36):
    k = 0
    l = 0
    for i in range(2895):
        if np.argmax(y_batch[i]) == j:
            k = k + 1
            if Y[i] == j:
                if np.argmax(y_batch[i]) == Y[i]:
                    l = l + 1
    if k == 0:
        continue
    rec = label_str[co]
    co = co+1
    print('%s召回率为%g' % (rec, (l / k)))

