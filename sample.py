import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#回帰部分の実装
x = tf.placeholder(tf.float32, [None, 784])#画像(784次元)
W = tf.Variable(tf.zeros([784, 10]))#784次元に相当する重み生成,10種類の確率変数を生成(i)
b = tf.Variable(tf.zeros([10]))#10種類の重みを生成(i)
y = tf.nn.softmax(tf.matmul(x, W) + b)

#訓練
y_ = tf.placeholder(tf.float32, [None, 10])#交差エントロピー 関数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#勾配降下法
sess = tf.InteractiveSession()#launch
tf.global_variables_initializer().run()
for i in range(1000):
  print("looping:",i)
  batch_xs, batch_ys = mnist.train.next_batch(250)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#評価
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))