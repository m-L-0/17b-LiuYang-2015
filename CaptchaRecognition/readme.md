# CaptchaRecognition
## 数据统计结果:
![img_0](https://github.com/m-L-0/17b-LiuYang-2015/blob/master/CaptchaRecognition/images/index.png)
![img_1](https://github.com/m-L-0/17b-LiuYang-2015/blob/master/CaptchaRecognition/images/index0.png)

## 将数据集制作成TFRecord文件。
已经按照要求完成对训练集、验证集、测试集的划分,本次作业中采用8:1:1的分配比例完成样本分配,


## 设计模型
使用了卷积神经网络；

网络结构为:

C1层:32个3x3x1的卷积核,学习11种特征

S1层:2x2最大池化

C2层:64个3x3x32卷积核

S2层:2x2最大池化

C3层:96个3x3x64卷积核

S3层:2x2最大池化

Flatten层:5x6x96维度样本

4个输出层

优化器:AdamOptimizer

第一层卷积: 3×3×1卷积核32个 [3，3，1，32],conv1.shape=[-1, 36, 48, 32],学习11种特征

但在训练和验证过程中正确率差异较大,最终验证集合正确率约为0.884

## 尚未完成的部分
在设计网页API和TensorBoard可视化中遇到了问题,正在尝试解决,会尽快完成.
