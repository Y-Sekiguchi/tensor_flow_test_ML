### import tensorflow
import tensorflow as tf

### download MNIST dataset
print("download MNIST dataset")
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
print("finished")

### make array for image vectors
x=tf.placeholder(tf.float32, [None,784])

### make array for weight matrix
W=tf.Variable(tf.zeros([784,10]))

### make array for bias vector
b=tf.Variable(tf.zeros([10]))

### formulation for the model
y=tf.nn.softmax(tf.matmul(x,W)+b)

### make array for correct label
y_=tf.placeholder(tf.float32, [None,10])

### define cross-entropy for cost function
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

### define training method
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

### executing training
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
print("start training")
for i in range(1000):
  batch_xs, batch_ys=mnist.train.next_batch(100)
  sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
print("end training")

### evaluate the model with trained parameters comparing with test data
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


