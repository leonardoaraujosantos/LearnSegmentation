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
FOLDER = '//mnt/fs3/QA_analitics/Person_Re_Identification/git_repo/datasets/COCO/'
TRAIN_ANOTATION = FOLDER + 'test2014/'
VALIDATION_ANOTATION = FOLDER + 'annotations/validation/'
TRAIN_IMAGES = FOLDER + 'test2014/'
VALIDATION_IMAGES = FOLDER + 'images/validation/'
CLASS_FILE = FOLDER + 'objectInfo150.txt'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LMDB_PATH = '//mnt/fs3/QA_analitics/Person_Re_Identification/git_repo/datasets/COCO/COCO_LMDB_Test'

# ### Parse class text file



dataset = []

# Get list of all files inside directory TRAIN_IMAGES
list_images_train = os.listdir(TRAIN_IMAGES)

# Iterate the list, loading the image and matching label
for img_path in list_images_train:
    # Get the sample name
    img_path = TRAIN_IMAGES + img_path

    # Read images and resize
    try:
        img = misc.imread(img_path)
        img = misc.imresize(img, [IMAGE_HEIGHT, IMAGE_WIDTH], interp='nearest')

    except:
        continue

    # img = misc.imread(img_path)


    if img.shape != (IMAGE_WIDTH, IMAGE_WIDTH, 3):
        print('Error image shape:', img_path, 'shape:', img.shape)
        continue


    # Append image and label tupples to the dataset list
    dataset.append((img))

# ### Shuffle dataset

# In[ ]:

random.shuffle(dataset)
print('Dataset size:', len(dataset))

# ### Calculate sizes

# In[ ]:

# get size in bytes of lists
size_bytes_images = dataset[0].nbytes * len(dataset)

total_size = size_bytes_images
print('Total size(bytes): %d' % (size_bytes_images ))

# ### Create LMDB File

# In[ ]:

# Open LMDB file
# You can append more information up to the total size.
env = lmdb.open(LMDB_PATH, map_size=total_size * 30)

# ### Add information on LMDB file

# In[ ]:

# Counter to keep track of LMDB index
idx_lmdb = 0
# Get a write lmdb transaction, lmdb store stuff with a key,value(in bytes) format
with env.begin(write=True) as txn:
    # Iterate on batch
    for (tup_element) in dataset:
        img = tup_element

        # Get image shapes
        shape_str_img = '_'.join([str(dim) for dim in img.shape])

        # Encode shape information on key
        img_id = ('img_{:08}_' + shape_str_img).format(idx_lmdb)
        # Put data
        txn.put(bytes(img_id.encode('ascii')), img.tobytes())
        idx_lmdb += 1



