"""P2 - Traffic Sign Classifier - Keras implementation"""
import pickle
import preprocessor
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelBinarizer

tf.python.control_flow_ops = tf

training_file = './data/augmented_train.p'
validation_file = './data/valid.p'
testing_file = './data/test.p'

# Load training and validation data
with open(training_file, mode='rb') as f:
    data = pickle.load(f)

with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

X_train, y_train = data['features'], data['labels']
X_valid, y_valid = valid['features'], valid['labels']

# Input
input_layer = Input(shape=(32, 32, 1))

# Layer 1
layer1 = Convolution2D(16, 5, 5, border_mode='valid')(input_layer)
layer1 = Activation('relu')(layer1)
layer1 = MaxPooling2D((2, 2))(layer1)

# Layer 2
layer2 = Convolution2D(32, 5, 5, border_mode='valid')(layer1)
layer2 = Activation('relu')(layer2)
layer2 = MaxPooling2D((2, 2))(layer2)

# Layer 3
layer3 = Convolution2D(512, 5, 5, border_mode='valid')(layer2)
layer3 = Activation('relu')(layer3)

# Flatten layers
layer1_flat = Flatten()(layer1)
layer2_flat = Flatten()(layer2)
layer3_flat = Flatten()(layer3)

flattened = merge([layer1_flat, layer2_flat, layer3_flat], mode='concat')
flattened = Dropout(0.5)(flattened)

# Layer 4
layer4 = Dense(43)(flattened)
layer4 = Activation('softmax')(layer4)

model = Model(input=input_layer, output=layer4)

model.summary()

# Preprocess data
X_train_preprocessed = preprocessor.preprocess(X_train)
X_valid_preprocessed = preprocessor.preprocess(X_valid)

# Create logging setup
logs = TensorBoard(histogram_freq=5)

label_binarizer = LabelBinarizer()
y_train_one_hot = label_binarizer.fit_transform(y_train)
y_valid_one_hot = label_binarizer.fit_transform(y_valid)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train_preprocessed, y_train_one_hot, nb_epoch=5, validation_data=(X_valid_preprocessed, y_valid_one_hot), callbacks=[logs])


# Evaluate the models performance on test set
with open(testing_file, 'rb') as f:
    data_test = pickle.load(f)

X_test = data_test['features']
y_test = data_test['labels']

# preprocess data
X_test_preprocessed = preprocessor.preprocess(X_test)
y_test_one_hot = label_binarizer.fit_transform(y_test)

print("Testing")

metrics = model.evaluate(X_test_preprocessed, y_test_one_hot)

for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))


model.save('./models/lenet_keras.h5')

# Save model architecture
with open('./models/lenet_keras.yml', 'w') as f:
    f.write(model.to_yaml())

print("Model saved")
