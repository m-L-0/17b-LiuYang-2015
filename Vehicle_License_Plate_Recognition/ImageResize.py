import os        # os：操作系统相关的信息模块  
from PIL import Image
PATH = '1'
#/home/xiao...：绝对地址，/home/xiao...：相对地址;
data_base_dir = "车牌字符识别训练数据/数字/"+PATH     #存放原始图片地址  
save_dir = "saveddata"         #保存生成图片地址  start_neg_dir = 1
file_list = []      #建立新列表，用于存放图片名
for file in os.listdir(data_base_dir):   #指定目录：data_base_dir中内容
    file_list.append(file)     #将jpg图片文件全部全部存入file_list列表中
number_of_pictures = len(file_list)     #len(a):列表a长度  
n = 0
for i in file_list:
    img = Image.open(data_base_dir+'/'+i)
    n = n+1
    print(img)
    out = img.resize((20, 36), Image.ANTIALIAS)
    print('out', out)
    out.save('SavedData/'+str(n)+'_label_'+PATH+'.jpg')
print("number_of_pictures:", number_of_pictures)   # 输出图片个数  
print("CURRENT_PATH : ", PATH)
