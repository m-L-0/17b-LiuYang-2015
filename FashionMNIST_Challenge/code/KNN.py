import tensorflow as tf  
import numpy as np  


def loadMNIST():  
    from tensorflow.examples.tutorials.mnist import input_data  
    mnist = input_data.read_data_sets('data/fashion', one_hot=True)  
    mnist.train.next_batch(55000)
    return mnist  


def KNN(mnist, num):  
    train_x, train_y = mnist.train.next_batch(55000)
    test_x, test_y = mnist.train.next_batch(num)

    xtr = tf.placeholder(tf.float32, [None, 784])
    xte = tf.placeholder(tf.float32, [784])  
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtr, tf.negative(xte)),2), reduction_indices=1))

    pred = tf.argmin(distance, 0)  

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    right = 0
    f = open("data/untitled.txt",'w')    
    for i in range(num):  
        ansIndex = sess.run(pred, {xtr: train_x, xte: test_x[i, :]})  
        print('预测为 : ', np.argmax(train_y[ansIndex]))  
        print('实际为 : ', np.argmax(test_y[i]))
        f.write(str(np.argmax(train_y[ansIndex])))
        f.write('\n')        
        if np.argmax(test_y[i]) == np.argmax(train_y[ansIndex]):  
            right += 1.0  
    accracy = right/num*1.0  
    print('正确率 : ', accracy*100, '%')


if __name__ == "__main__":  
    mnist = loadMNIST()
    KNN(mnist, 300)