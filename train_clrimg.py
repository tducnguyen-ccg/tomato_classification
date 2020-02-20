import tensorflow as tf
import numpy as np
import random
import glob
import cv2

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 100
display_step = 10

# Network Parameters
img_high = 128
img_width = 128
input_channel = 3
num_input = img_high*img_width
num_classes = 3
dropout = 0.75

print('Read images from folder and create labels')
classes = {'Orange': 0, 'Yellow': 1, 'Maroon': 2}
folders = ['training_clr', 'testing_clr']
data_pth = '/home/tducnguyen/NguyenTran/Study/AIHot/1_Tomato_classification/Code/data/'
# Create list file test
train_images = []
test_images = []
train_labels = []
test_labels = []
for fd in folders:
    for cl in list(classes.keys()):
        img_paths = glob.glob(data_pth + fd + '/' + cl + '/*.jpg')
        for img_path in img_paths:
            if fd == 'training_clr':
                train_images.append(img_path)
                train_labels.append(int(classes[cl]))
            else:
                test_images.append(img_path)
                test_labels.append(int(classes[cl]))

# tf Graph input
X = tf.placeholder(tf.float32, [None, img_width, img_high, input_channel])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def next_batch(data, label, batch_sz, shuffle=True):
    if shuffle:
        index = random.sample(range(len(data)), batch_sz)
    else:
        index = list(range(len(data)))

    batch_data = []
    batch_label = []
    for id, i in enumerate(index):
        img_pth = data[i]
        img = np.asarray(cv2.resize(cv2.imread(img_pth), (img_width, img_high)))
        # img = np.reshape(img, [-1, img_width, img_high, input_channel])
        # if id == 0:
        #     batch_data = img
        # else:
        #     batch_data = np.concatenate((batch_data, img), axis=0)
        batch_data.append(img)

        img_label = np.zeros(3)
        img_label[label[i]] = 1
        batch_label.append(img_label)

    return batch_data, batch_label


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, img_high, img_width, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=4)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
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
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 5x5 conv, 64 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([64])),
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
        batch_x, batch_y = next_batch(train_images, train_labels, batch_size)
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

    test_x, test_y = next_batch(test_images, test_labels, batch_size,  shuffle=False)
    print("Testing Accuracy:",\
        sess.run(accuracy, feed_dict={X: test_x,
                                      Y: test_y,
                                      keep_prob: 1.0}))