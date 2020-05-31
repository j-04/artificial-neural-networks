import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_train = mnist.train.num_examples
n_validation = mnist.validation.num_examples
n_test = mnist.test.num_examples

n_input = 784
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_output = 10

learning_rate = 1e-4
n_iterations = 10000
batch_size = 128
dropout = 0.5

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'w1' : tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev = 0.1)),
    'w2' : tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev = 0.1)),
    'w3' : tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev = 0.1)),
    'out' : tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev = 0.1))
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) +biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict = {
        x: batch_x, 
        y: batch_y, 
        keep_prob:dropout
    })

    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy],
        feed_dict={
            x: batch_x,
            y: batch_y,
            keep_prob: 1.0
        })
        print ("Iteration", 
        str(i), 
        "\t| Loss = ",
        str(minibatch_loss),
        "\t| Accuracy = ",
        str(minibatch_accuracy))

test_accuracy = sess.run(accuracy, feed_dict = {
    x: mnist.test.images,
    y: mnist.test.labels,
    keep_prob: 1.0
})
print("\nAccuracy on test set: ", test_accuracy)

def recognize_digit(file_path):
    img = np.invert(Image.open(file_path).convert('L')).ravel()
    prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={x: [img]})
    print("Prediciton for test image: ", np.squeeze(prediction))

recognize_digit("number1.png")
recognize_digit("number2.png")
recognize_digit("number3.png")
recognize_digit("number8.png")
