import os

# 需要批量改名的文件所在文件夹
path_name='../../data/raw/乔巴/'

# 命名从1开始
i = 1
for item in os.listdir(path_name):
    # 进入到文件夹内，对每个文件进行循环遍历
    # os.path.join(path_name, item)表示找到每个文件的绝对路径并进行拼接操作
    os.rename(os.path.join(path_name, item), os.path.join(path_name, (str(i) + '.png')))
    i += 1
