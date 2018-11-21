# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function   
import tensorflow as tf
import numpy as np

# create data 这是一个基准函数
x_data = np.random.rand(100).astype(np.float32) #大部分的数据都是float32格式
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ### 下面是定义一个预测的函数，通过ML不断接近基准函数
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #定义数的range是-1～1
biases = tf.Variable(tf.zeros([1])) #biases的初始值定义为0

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) #计算预测的y与基准的y的差值
optimizer = tf.train.GradientDescentOptimizer(0.5) #用optimizer（优化器）减少误差；GradientDescentOptimizer（）基础的optimizer。括号里是学习效率，一般小于1
train = optimizer.minimize(loss) #
### create tensorflow structure end ###

sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init) #激活init ***非常重要

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


