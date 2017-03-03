"""P2 - Traffic Sign Classifier"""
import numpy as np
import pickle
import preprocessor
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Load the data
training_file = './data/augmented_train.p'
validation_file = './data/valid.p'
testing_file = './data/test.p'
logs_path = './logs'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = X_train.shape[0]
n_test = X_test.shape[0]
image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
n_classes = len(set(y_train))
mean = np.mean(X_train)
std = np.std(X_train)


# preprocess images
X_train_preprocessed = preprocessor.preprocess(X_train)
X_valid_preprocessed = preprocessor.preprocess(X_valid)
X_test_preprocessed = preprocessor.preprocess(X_test)

# Hyperparameters
epochs = 50
batch_size = 100
learning_rate = 0.001
mu = 0
sigma = 0.1

# Data placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

# LAYER 1
with tf.name_scope('Layer1'):
    layer1_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean=mu, stddev=sigma))
    layer1_biases = tf.Variable(tf.zeros(16))
    layer1 = tf.nn.conv2d(x, layer1_weights, strides=[1, 1, 1, 1], padding='VALID') + layer1_biases
    layer1 = tf.nn.relu(layer1)

    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# LAYER 2
with tf.name_scope('Layer2'):
    layer2_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma))
    layer2_biases = tf.Variable(tf.zeros(32))
    layer2 = tf.nn.conv2d(layer1, layer2_weights, strides=[1, 1, 1, 1], padding='VALID') + layer2_biases
    layer2 = tf.nn.relu(layer2)

    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# LAYER 3
with tf.name_scope('Layer3'):
    layer3_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 512), mean=mu, stddev=sigma))
    layer3_biases = tf.Variable(tf.zeros(512))
    layer3 = tf.nn.conv2d(layer2, layer3_weights, strides=[1, 1, 1, 1], padding='VALID') + layer3_biases
    layer3 = tf.nn.relu(layer3)


# Flatten layers
with tf.name_scope('Flatten'):
    layer1_flat = flatten(layer1)
    layer2_flat = flatten(layer2)
    layer3_flat = flatten(layer3)
    flattened = tf.concat_v2([layer1_flat, layer2_flat, layer3_flat], 1)

with tf.name_scope('Dropout'):
    flattened_shape = flattened.get_shape().as_list()
    flattened = tf.nn.dropout(flattened, keep_prob)


# LAYER 4
with tf.name_scope('Layer4'):
    layer4_weights = tf.Variable(tf.truncated_normal(shape=(flattened_shape[1], 43), mean=mu, stddev=sigma))
    layer4_biases = tf.Variable(tf.zeros(43))
    logits = tf.matmul(flattened, layer4_weights) + layer4_biases

labels_predicted = tf.nn.softmax(logits)

one_hot_y = tf.one_hot(y, 43)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


# Create data logging functionality
tf.summary.scalar("cost", loss_operation)
tf.summary.scalar("accuracy", accuracy_operation)
summary_operation = tf.merge_all_summaries()


def evaluate(X_data, y_data):
    """Evaluate model."""
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Start the tensorflow session and run the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    log_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    num_examples = len(X_train_preprocessed)

    print("Training...")
    print()
    for i in range(epochs):
        X_train_shuffled, y_train_shuffled = shuffle(X_train_preprocessed, y_train)

        # number of batches in one epoch
        batch_count = int(num_examples / batch_size)

        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train_shuffled[offset:end], y_train_shuffled[offset:end]
            _, summary = sess.run([training_operation, summary_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            # Write log
            log_writer.add_summary(summary, i * batch_count + int(offset / batch_size))

        validation_accuracy = evaluate(X_valid_preprocessed, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    test_accuracy = evaluate(X_test_preprocessed, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    saver.save(sess, './models/lenet')
    print("Model saved")
