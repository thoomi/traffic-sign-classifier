"""Script to generate augmented training data"""
import cv2
import numpy as np
import pickle
import random
from collections import Counter


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


def image_shear(img, shear_range):
    """Shear image randomly by the factor of shear_range"""
    rows, cols, dims = img.shape

    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    matrix = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(img, matrix, (cols, rows))


def random_image_transform(image):
    """Transform image according to given parameters"""
    randomAngle = random.randint(-10, 10)
    output = image_rotate(image, randomAngle)

    randomX = random.randint(-2, 2)
    randomY = random.randint(-2, 2)
    output = image_translate(output, randomX, randomY)

    randomShear = random.randint(-5, 5)
    output = image_shear(output, randomShear)

    return output


# Load original training data
training_file = './data/train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

n_classes = len(set(y_train))


# Generate additional images
max_images_per_class = 2500
new_images = []
new_labels = []

labels, count_signs_by_class = zip(*Counter(y_train).items())
count_per_class = np.array(count_signs_by_class)

# Generate additional images until every class has the same number of images
all_images_created = False

print('Generating additional images...')

while not all_images_created:
    for index, image in enumerate(X_train):
        img_class = y_train[index]

        if count_per_class[img_class] < 2500:
            new_images.append(random_image_transform(image))
            new_labels.append(img_class)
            count_per_class[img_class] += 1

    if np.sum(count_per_class) >= n_classes * max_images_per_class:
        all_images_created = True

X_train_generated = np.append(X_train, new_images, axis=0)
y_train_generated = np.append(y_train, new_labels, axis=0)


# Save augumented data
save_file = './data/augmented_train.p'
augumented_data = {'features': X_train_generated, 'labels': y_train_generated}
pickle.dump(augumented_data, open(save_file, 'wb'))

print('Saved new training data to: ', save_file)
