import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,activation_function = None):  #in_size  out_size is the nums of the cell
    #add one more layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) +0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    #stride[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME") #Must have stride[0]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")

#define placeholder for the input to network
xs = tf.placeholder(tf.float32,[None,784])   #any sample of the data has 784 dim data
ys = tf.placeholder(tf.float32,[None,10])

#add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)


#caculate the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#run the tensorflow
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 ==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
