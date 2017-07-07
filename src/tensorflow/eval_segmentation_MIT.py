import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.misc as misc
import os
import scipy.misc as misc
from natsort import natsorted
import utils_eval
import config

import sys

sys.path.insert(0, '../tensorflow')
import models

MODEL_FOLDER = '/opt/home/giounona/LearnSegmentation/src/tensorflow/save/model-200'

IMAGE_FOLDER = '/mnt/fs3/QA_analitics/Person_Re_Identification/git_repo/datasets/MITseg/ADEChallengeData2016'

VALIDATION_ANOTATION = IMAGE_FOLDER + 'annotations/validation/'

VALIDATION_IMAGES = IMAGE_FOLDER + 'images/validation/'

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


# IMAGE_TEST = '/media/laraujo/BigLinuxPart/Open Datasets/sceneparsing/ADEChallengeData2016/images/training/ADE_train_00014160.jpg'
# IMAGE_GT = '/media/laraujo/BigLinuxPart/Open Datasets/sceneparsing/ADEChallengeData2016/annotations/training/ADE_train_00014160.png'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASS_NUM = 8
IMAGE_NUM = 4000


def color_mask(label, color, idx):
    mask = np.zeros((label.shape[0], label.shape[1]))
    k = np.tile(color, (label.shape[0], label.shape[1], 1))
    indices = np.where(np.all(label == color, axis=-1))
    mask[indices] = idx
    return mask


pixel_accuracy = np.zeros([IMAGE_NUM])
pixel_correct = np.zeros([IMAGE_NUM])
pixel_labeled = np.zeros([IMAGE_NUM])
area_intersection = np.zeros([CLASS_NUM, IMAGE_NUM])
area_union = np.zeros([CLASS_NUM, IMAGE_NUM])

# Parameters
gpu = 0
segmentation_type = 'segnet'
#
# fig = plt.figure()
# a=fig.add_subplot(1,2,1)
# img_input = misc.imread(IMAGE_TEST)
# img_label = misc.imread(IMAGE_GT)
# plt.imshow(img_input)
# a.set_title('Input')
# a=fig.add_subplot(1,2,2)
# #plt.imshow(img_label, cmap=cm.Paired ,vmin=np.min(img_label), vmax=np.max(img_label))
# plt.imshow(img_label, cmap=cm.Paired ,vmin=0, vmax=150)
# #plt.imshow(img_label)
# a.set_title('Label')
# plt.colorbar()
# plt.show()

if gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
else:
    print('Set tensorflow on CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Build model
if segmentation_type.lower() == 'segnet':
    segmentation_model = models.SegnetNoConnected(training_mode=False)
elif segmentation_type.lower() == 'segnet_connected':
    segmentation_model = models.SegnetConnected(training_mode=False)
elif segmentation_type.lower() == 'segnet_connected_gate':
    segmentation_model = models.SegnetConnectedGate(training_mode=False)
else:
    segmentation_model = models.FullyConvolutionalNetworks(training_mode=False)

# Get Placeholders
model_in = segmentation_model.input
model_out = segmentation_model.output
anotation_prediction = segmentation_model.anotation_prediction

# Load tensorflow model
print("Loading model: %s" % MODEL_FOLDER)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, MODEL_FOLDER)

count = 0


# Get list of all files inside directory TRAIN_IMAGES
list_images_train = os.listdir(VALIDATION_IMAGES)


# Iterate the list, loading the image and matching label
for img_path in list_images_train:
    # Get the sample name
    sample = img_path.split('.')[0]
    label_name = sample + '.png'
    img_path = VALIDATION_IMAGES + img_path
    label_path = VALIDATION_ANOTATION + label_name

    # Read images and resize
    img = misc.imread(img_path)
    label = misc.imread(label_path)
    img = misc.imresize(img, [IMAGE_HEIGHT, IMAGE_WIDTH], interp='nearest')
    label = misc.imresize(label, [IMAGE_HEIGHT, IMAGE_WIDTH], interp='nearest')

    pred_to_eval = anotation_prediction.eval(feed_dict={model_in: [img]})[0]

    fig = plt.figure()
    a=fig.add_subplot(1,3,1)
    plt.imshow(img)
    a.set_title('Input')
    a=fig.add_subplot(1,3,2)
    plt.imshow(label, cmap=cm.Paired ,vmin=0, vmax=150)
    a = fig.add_subplot(1, 3, 3)
    plt.imshow(pred_to_eval, cmap=cm.Paired ,vmin=0, vmax=150)
    a.set_title('Prediction')


    plt.colorbar()
    plt.show()


    (pixel_accuracy[count], pixel_correct[count], pixel_labeled[count]) = utils_eval.pixelAccuracy(pred_to_eval, label)
    (area_intersection[:, count], area_union[:, count]) = utils_eval.intersectionAndUnion(pred_to_eval, label,
                                                                                                  CLASS_NUM)
    count = count + 1
#
#
IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1) + area_union, axis=1)

mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
#
print("IoU: %f" % np.mean(IoU))
print("Mean pixel accuracy: %f" % mean_pixel_accuracy)


