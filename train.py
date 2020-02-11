import tensorflow.compat.v1 as tf
import numpy as np
import random
tf.disable_v2_behavior()


# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

# Network Parameters
num_input = 512
num_classes = 3
dropout = 0.75

train_path = 'data/training/label/label.txt'
test_path = 'data/testing/label/label.txt'
with open(train_path, 'r') as f:
    train_data = f.read()
    train_data = train_data.split('\n')
with open(test_path, 'r') as f:
    test_data = f.read()
    test_data = test_data.split('\n')

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def next_batch(data, batch_sz, shuffle=True):
    if shuffle:
        index = random.sample(range(len(data)), batch_sz)
    else:
        index = list(range(len(data)))

    batch_data = []
    batch_label = []
    for i in index:
        img_name = data[i].split(' ')[0]
        img_label = np.zeros(3)
        try:
            img_label[int(data[i].split(' ')[1])] = 1
        except:
            print()

        img_feature = np.load(img_name)
        img_feature = np.reshape(img_feature, len(img_feature))

        batch_data.append(img_feature)
        batch_label.append(img_label)

    return batch_data, batch_label


# Create some wrappers for simplicity
def conv1d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv1d(x, W, stride=1, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool1d(x, k=2):
    # MaxPool1D wrapper
    return tf.nn.max_pool1d(x, ksize=2, strides=2, padding='VALID')


# Create model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 512, 1])

    # Convolution Layer
    conv1 = conv1d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool1d(conv1, k=2)

    # Convolution Layer
    conv2 = conv1d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool1d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([128*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch(train_data, batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    test_x, test_y = next_batch(test_data, batch_size,  shuffle=False)
    print("Testing Accuracy:",\
        sess.run(accuracy, feed_dict={X: test_x,
                                      Y: test_y,
                                      keep_prob: 1.0}))