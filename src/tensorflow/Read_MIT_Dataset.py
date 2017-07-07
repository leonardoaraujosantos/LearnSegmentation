# coding: utf-8

# ## Create LMDB from Mit Scene parsing
# On this notebook we will create a LMDB dataset from the MIT Scene Parsing data. On this dataset there are 150 classes, the information os organized as follows:
# * Image files (Train and Validation)
# * Anotation (Train and Validation)
# * Text file describing classes
#
#
# ### References
# * http://sceneparsing.csail.mit.edu/results2016.html

# In[1]:

import lmdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
from os import walk
import scipy.misc as misc

# Some constant definitions
FOLDER = '/home/giounona/git_repo/datasets/MITseg/ADEChallengeData2016/'
TRAIN_ANOTATION = FOLDER + 'annotations/training/'
VALIDATION_ANOTATION = FOLDER + 'annotations/validation/'
TRAIN_IMAGES = FOLDER + 'images/training/'
VALIDATION_IMAGES = FOLDER + 'images/validation/'
CLASS_FILE = FOLDER + 'objectInfo150.txt'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LMDB_PATH = '/home/giounona/git_repo/datasets/MITseg/Segmentation_LMDB_Train'

# ### Parse class text file

# In[2]:

data = pd.read_csv(CLASS_FILE, sep="\t", header=0)
data.columns = ["Idx", "Ratio", "Train", "Validation", "Name"]
data.head()

# ### Display input/label train images
# Observe that the key to find the match between input image and label is the filename. Also observe that the label images has png extension.

# In[3]:

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_input = mpimg.imread(TRAIN_IMAGES + 'ADE_train_00000007.jpg')
img_label = mpimg.imread(TRAIN_ANOTATION + 'ADE_train_00000007.png')
plt.imshow(img_input)
a.set_title('Input')
a = fig.add_subplot(1, 2, 2)
plt.imshow(img_label)
a.set_title('Label')
plt.show()

# ### Create list of path to every image on train/label
# Also we load both image input and label to a list

# In[7]:

dataset = []

# Get list of all files inside directory TRAIN_IMAGES
list_images_train = os.listdir(TRAIN_IMAGES)

# Iterate the list, loading the image and matching label
for img_path in list_images_train:
    # Get the sample name
    sample = img_path.split('.')[0]
    label_name = sample + '.png'
    img_path = TRAIN_IMAGES + img_path
    label_path = TRAIN_ANOTATION + label_name

    # Read images and resize
    img = misc.imread(img_path)
    label = misc.imread(label_path)
    img = misc.imresize(img, [IMAGE_HEIGHT, IMAGE_WIDTH], interp='nearest')
    label = misc.imresize(label, [IMAGE_HEIGHT, IMAGE_WIDTH], interp='nearest')

    if img.shape != (IMAGE_WIDTH, IMAGE_WIDTH, 3):
        print('Error image shape:', img_path, 'shape:', img.shape)
        continue

    if label.shape != (IMAGE_WIDTH, IMAGE_WIDTH):
        print('Error label shape:', label_path, 'shape:', label.shape)
        continue

    # Append image and label tupples to the dataset list
    dataset.append((img, label))

# ### Shuffle dataset

# In[ ]:

random.shuffle(dataset)
print('Dataset size:', len(dataset))

# ### Calculate sizes

# In[ ]:

# get size in bytes of lists
size_bytes_images = dataset[0][0].nbytes * len(dataset)
size_bytes_labels = dataset[0][1].nbytes * len(dataset)
total_size = size_bytes_images + size_bytes_labels
print('Total size(bytes): %d' % (size_bytes_images + size_bytes_labels))

# ### Create LMDB File

# In[ ]:

# Open LMDB file
# You can append more information up to the total size.
env = lmdb.open(LMDB_PATH, map_size=total_size * 15)

# ### Add information on LMDB file

# In[ ]:

# Counter to keep track of LMDB index
idx_lmdb = 0
# Get a write lmdb transaction, lmdb store stuff with a key,value(in bytes) format
with env.begin(write=True) as txn:
    # Iterate on batch
    for (tup_element) in dataset:
        img, label = tup_element

        # Get image shapes
        shape_str_img = '_'.join([str(dim) for dim in img.shape])
        shape_str_label = '_'.join([str(dim) for dim in label.shape])

        label_id = ('label_{:08}_' + shape_str_label).format(idx_lmdb)
        # Encode shape information on key
        img_id = ('img_{:08}_' + shape_str_img).format(idx_lmdb)
        # Put data
        txn.put(bytes(label_id.encode('ascii')), label.tobytes())
        txn.put(bytes(img_id.encode('ascii')), img.tobytes())
        idx_lmdb += 1


# In[ ]:



