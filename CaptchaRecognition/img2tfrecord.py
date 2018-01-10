import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import csv


# 生成整型属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


img_list = []
reader = csv.reader(open('data/captcha/labels/labels.csv', encoding='utf-8'))
for i in reader:
    img_list.append(i[1])

cwd = '/home/miles/CaptchaRecognition/data/captcha/images/'
writer0 = tf.python_io.TFRecordWriter("captcha_train_0.tfrecords")  # 生成训练集
writer1 = tf.python_io.TFRecordWriter("captcha_train_1.tfrecords")  # 生成训练集
writer2 = tf.python_io.TFRecordWriter("captcha_train_2.tfrecords")  # 生成训练集
writer3 = tf.python_io.TFRecordWriter("captcha_train_3.tfrecords")  # 生成训练集
writer4 = tf.python_io.TFRecordWriter("captcha_train_4.tfrecords")  # 生成训练集
writer5 = tf.python_io.TFRecordWriter("captcha_train_5.tfrecords")  # 生成训练集
writer6 = tf.python_io.TFRecordWriter("captcha_train_6.tfrecords")  # 生成训练集
writer7 = tf.python_io.TFRecordWriter("captcha_train_7.tfrecords")  # 生成训练集
writer8 = tf.python_io.TFRecordWriter(
    "captcha_validation_0.tfrecords")  # 生成验证集
writer9 = tf.python_io.TFRecordWriter("captcha_test_0.tfrecords")  # 生成测试集
addr = 0
for i in range(0, 39999):
    addr = addr + 1
    img_path = cwd + str(i) + '.jpg'
    img = Image.open(img_path)
    img = img.resize((54, 38))
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': _int64_feature(int(img_list[i])),
            'image_raw': _bytes_feature(img_raw)
        }))
    print(i)
    print(img_list[i])
    if (i < 4000):
        writer0.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 8000):
        writer1.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 12000):
        writer2.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 16000):
        writer3.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 20000):
        writer4.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 24000):
        writer5.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 28000):
        writer6.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 32000):
        writer7.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 36000):
        writer8.write(example.SerializeToString())  # 序列化为字符串
    elif (i < 40000):
        writer9.write(example.SerializeToString())  # 序列化为字符串
writer0.close()
writer1.close()
writer2.close()
writer3.close()
writer4.close()
writer5.close()
writer6.close()
writer7.close()
writer8.close()
writer9.close()