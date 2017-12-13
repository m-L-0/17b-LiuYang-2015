import os
import matplotlib.pyplot as plt

cwd = '车牌字符识别训练数据(copy)/数字+字母'
classes = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
}  # 人为 设定 34 类
file_list = []
data = []
label = []
for index, name in enumerate(classes):
    class_path = cwd + '/' + name + '/'
    addr = 0
    for file in os.listdir(class_path):  # 指定目录：data_base_dir中内容
        file_list.append(file)  # 将jpg图片文件全部全部存入file_list列表中
    label.append(len(file_list))
    data.append(name)
    file_list = []


patches, l_text, p_text = plt.pie(
    label,
    labels=data,
    labeldistance=1.1,
    autopct='%2.0f%%',
    shadow=False,
    startangle=90,
    pctdistance=0.6)
# labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
# autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
# shadow，饼是否有阴影
# startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
# pctdistance，百分比的text离圆心的距离
# patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本

# 改变文本的大小
# 方法是把每一个text遍历。调用set_size方法设置它的属性
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20
# 设置x，y轴刻度一致，这样饼图才能是圆的
plt.axis('equal')
plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
# loc: 表示legend的位置，包括'upper right','upper left','lower right','lower left'等
# bbox_to_anchor: 表示legend距离图形之间的距离，当出现图形与legend重叠时，可使用bbox_to_anchor进行调整legend的位置
# 由两个参数决定，第一个参数为legend距离左边的距离，第二个参数为距离下面的距离
plt.grid()
plt.show()

plt.bar(range(len(data)), label, tick_label=data)
plt.show()
