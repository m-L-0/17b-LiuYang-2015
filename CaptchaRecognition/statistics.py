import matplotlib.pyplot as plt
import csv

img_list = []
count = count0 = count1 = count2 = count3 = 0
reader = csv.reader(open('data/captcha/labels/labels.csv', encoding='utf-8'))
for i in reader:
    img_list.append(i[1])
    if int(i[1]) > 999:
        count3 += 1
    elif int(i[1]) > 99:
        count2 += 1
    elif int(i[1]) > 9:
        count1 += 1
    else:
        count0 += 1
    count += 1
print(count)
print(count0)
print(count1)
print(count2)
print(count3)
name_list = [
    '1-digit captcha', '2-digit captcha', '3-digit captcha', '4-digit captcha'
]
label_list = [count0, count1, count2, count3]

patches, l_text, p_text = plt.pie(
    label_list,
    labels=name_list,
    labeldistance=1.1,
    autopct='%2.0f%%',
    shadow=False,
    startangle=90,
    pctdistance=0.6)
for t in l_text:
    t.set_size = 30
for t in p_text:
    t.set_size = 20
plt.axis('equal')
plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
plt.grid()
plt.show()
plt.bar(range(len(name_list)), label_list, tick_label=name_list)
plt.show()