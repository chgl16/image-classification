from skimage import io,transform
import tensorflow as tf
import numpy as np
import glob

path = '../../data/test-set/*.png'

image_dict = {0: '路飞',1:'罗宾',2:'娜美',3:'乔巴',4:'索隆'}

w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
  
    # 目录列表
    paths = glob.glob(path)
    for img in paths:
        data.append(read_one_image(img))

    saver = tf.train.import_meta_graph('../../model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('../../model/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)
    

    #打印出预测矩阵
    print("\n预测矩阵:\n", classification_result)
    #打印出预测矩阵每一行最大值的索引
    print("\n简略结果:\n", tf.argmax(classification_result,1).eval(), '\n')
    print("具体情况: ")
    #根据索引通过字典对应人物的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    count = 0
    for i in range(len(output)):
        # output[i]是测试结果编码，paths[i])[-7]是原定图片编号（路飞1）
        flag = False
        if str(output[i]+1) == paths[i][-7]:
            flag = True
            count += 1
        print("第 " + str(i+1) +  " 张 (" +   paths[i][-7:] + ") 人物预测: " + image_dict[output[i]]  + " " + str(flag))
    print("\n准确率: {:.2f}%".format(count / len(output) * 100 ))


