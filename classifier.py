"""P2 - Traffic Sign Classifier"""
import cv2
import numpy as np
import pickle
import random
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Load the data
training_file = './data/train.p'
validation_file = './data/valid.p'
testing_file = './data/test.p'

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


def center_normalize(data, mean, std):
    """Center normalize images"""
    data = data.astype('float32')
    data -= mean
    data /= std
    return data


def preprocess(data):
    """Convert to grayscale, histogram equalize, and expand dims"""
    imgs = np.ndarray((data.shape[0], 32, 32, 1), dtype=np.uint8)
    for i, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = np.expand_dims(img, axis=2)
        imgs[i] = img

    imgs = center_normalize(imgs, mean, std)
    return imgs


def image_rotate(img, angle):
    """Rotate image by angle"""
    rows, cols, dims = img.shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, matrix, (cols, rows))


def image_translate(img, x, y):
    """Translate image by the value of x and y"""
    rows, cols, dims = img.shape
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, matrix, (cols, rows))


def random_image_transform(image):
    """Transform image according to given parameters"""
    randomAngle = random.randint(-10, 10)
    output = image_rotate(image, randomAngle)

    randomX = random.randint(-2, 2)
    randomY = random.randint(-2, 2)
    output = image_translate(output, randomX, randomY)

    return output


# Generate additional images
additional_images = 4
new_images = []
new_labels = []

for indexOfImage, img in enumerate(X_train):
    # Generate five additional images for each original image
    for i in range(additional_images):
        new_images.append(random_image_transform(img))
        new_labels.append(y_train[indexOfImage])

X_train = np.append(X_train, new_images, axis=0)
y_train = np.append(y_train, new_labels, axis=0)

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test = preprocess(X_test)

# Hyperparameters
epochs = 60
batch_size = 100
learning_rate = 0.001
mu = 0
sigma = 0.1

# Data placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

# LAYER 1
layer1_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean=mu, stddev=sigma))
layer1_biases = tf.Variable(tf.zeros(16))
layer1 = tf.nn.conv2d(x, layer1_weights, strides=[1, 1, 1, 1], padding='VALID') + layer1_biases
layer1 = tf.nn.relu(layer1)

layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# LAYER 2
layer2_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma))
layer2_biases = tf.Variable(tf.zeros(32))
layer2 = tf.nn.conv2d(layer1, layer2_weights, strides=[1, 1, 1, 1], padding='VALID') + layer2_biases
layer2 = tf.nn.relu(layer2)

layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# LAYER 6
layer6_weights = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 512), mean=mu, stddev=sigma))
layer6_biases = tf.Variable(tf.zeros(512))
layer6 = tf.nn.conv2d(layer2, layer6_weights, strides=[1, 1, 1, 1], padding='VALID') + layer6_biases
layer6 = tf.nn.relu(layer6)


layer1_flat = flatten(layer1)
layer2_flat = flatten(layer2)
layer6_flat = flatten(layer6)
flattened = tf.concat_v2([layer1_flat, layer2_flat, layer6_flat], 1)
flattened_shape = flattened.get_shape().as_list()

flattened = tf.nn.dropout(flattened, keep_prob)

# LAYER 3
layer3_weights = tf.Variable(tf.truncated_normal(shape=(flattened_shape[1], 43), mean=mu, stddev=sigma))
layer3_biases = tf.Variable(tf.zeros(43))
# layer3 = tf.matmul(flattened, layer3_weights) + layer3_biases
logits = tf.matmul(flattened, layer3_weights) + layer3_biases
# layer3 = tf.nn.relu(layer3)


# LAYER 4
# layer4_weights = tf.Variable(tf.truncated_normal(shape=(1024, 84), mean=mu, stddev=sigma))
# layer4_biases = tf.Variable(tf.zeros(84))
# layer4 = tf.matmul(layer3, layer4_weights) + layer4_biases
# layer4 = tf.nn.relu(layer4)

# LAYER 5
# layer5_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
# layer5_biases = tf.Variable(tf.zeros(43))
# logits = tf.matmul(layer4, layer5_weights) + layer5_biases


one_hot_y = tf.one_hot(y, 43)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


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
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    saver.save(sess, './models/lenet')
    print("Model saved")
