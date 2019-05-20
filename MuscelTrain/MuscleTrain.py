import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

def loadData(filename):
    filedata = open(filename,"rt")
    senordata = []
    while True:
        linedata = filedata.readline()
        if not linedata:
            break
        linedata = linedata.strip('\n')
        loadata = linedata.split(' ')
        senordata.append(loadata)
    senordata = np.array(senordata,np.float32)
    return senordata

def add_layer(input,in_size,out_size,activation_function = None):
    #neuarl cell
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biase = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input,Weights) + biase
    #neuarl cell
    #activative function
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    # activative function
    return output



traindat = loadData("./data/train.data")
testdat = loadData("./data/test.data")
testrow = testdat.shape[0]
x_test = testdat[:,:18]
y_test = (-(testdat[:, 18] - 1) / 2).reshape(testrow, 1)  # 1:shoot   0:free
rows = traindat.shape[0]
cols = traindat.shape[1]

x_data = traindat[:, :18]
y_data = (-(traindat[:, 18] - 1) / 2).reshape(rows, 1)  # 1:shoot   0:free

inputdata = tf.placeholder(tf.float32, [None, 18])  # any of the data has 18 dim
labeldata = tf.placeholder(tf.float32, [None, 1])

lay1 = add_layer(inputdata, 18, 10, activation_function=tf.nn.relu)
pre = add_layer(lay1, 10, 1, activation_function=tf.nn.sigmoid)

# loss = tf.reduce_sum(tf.square(pre - labeldata),reduction_indices=[1])
# loss = tf.reduce_mean(-tf.reduce_sum(tf.log(((pre*labeldata)+1)/2), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-labeldata * tf.log(tf.clip_by_value(pre, 1e-10, 1.0)))
trainstep = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(trainstep, feed_dict={inputdata: x_data, labeldata: y_data})
    if i % 50 == 0:
        lossdata = sess.run(cross_entropy, feed_dict={inputdata: x_data, labeldata: y_data})
        print("Now loss is {}".format(lossdata))
        prediction = sess.run(pre, feed_dict={inputdata: x_data, labeldata: y_data})
        #print("pre is :{} {}".format(prediction,prediction.shape))

        pretest = sess.run(pre, feed_dict={inputdata: x_test, labeldata: y_test})
        cnt = 0
        acc = np.in1d(pretest,y_test)
        for i in range(testrow):
            if (acc[i]):
                cnt = cnt + 1
        print("the test accuracy is {}".format(cnt / testrow))


        count = 0
        acc = np.in1d(prediction,y_data)

        for i in range(rows):
            if(acc[i]):
                count = count +1
        print("the train accuracy is {}".format(count/rows))