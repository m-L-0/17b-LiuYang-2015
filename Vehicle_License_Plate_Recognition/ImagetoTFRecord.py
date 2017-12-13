import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


cwd = '车牌字符识别训练数据(copy)/数字+字母'
classes = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
}  # 人为 设定 34 类
writer1 = tf.python_io.TFRecordWriter("plate_train.tfrecords")  # 训练集的位置
writer2 = tf.python_io.TFRecordWriter("plate_validation.tfrecords")  # 验证集的位置
file_list = []

for index, name in enumerate(classes):
    class_path = cwd + '/' + name + '/'
    addr = 0
    for file in os.listdir(class_path):   # 指定目录：data_base_dir中内容
        file_list.append(file)     # 将jpg图片文件全部全部存入file_list列表中
    print(len(file_list))
    for img_name in os.listdir(class_path):
        addr = addr+1
        img_path = class_path + img_name  # 每一个图片的地址
        img = Image.open(img_path)
        img = img.resize((24, 48), Image.ANTIALIAS)
        img.save('Data/' + str(index) + '_' + name + '_label.jpg')
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(
            feature={   
                'label': _int64_feature(index),
                'image_raw': _bytes_feature(img_raw)}))  # example对象对label和image数据进行封装
        if (addr < len(file_list)*0.8):
            writer1.write(example.SerializeToString())  # 序列化为字符串
        else:
            writer2.write(example.SerializeToString())
    file_list = []
    addr = 0
writer1.close()
writer2.close()