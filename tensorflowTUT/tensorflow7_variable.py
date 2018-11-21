# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

state = tf.Variable(0, name='counter') #Tensorflow的变量要用Variable来设置
#print(state.name)
one = tf.constant(1) #常量1

new_value = tf.add(state, one) #常量加1个变量还是变量
update = tf.assign(state, new_value)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:    #打开session的一种模式，这种模式不需要sess.close()它会自动关闭
    sess.run(init)            #一定要run（init），不然你的variables就没初始化
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

