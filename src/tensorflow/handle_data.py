import scipy.misc
import random
import h5py
import lmdb
import numpy as np
import tensorflow as tf
from augment_batch import AugmentBatch

class HandleData:
    __xs = []
    __ys = []
    __file = []
    __file_val = []
    __dataset_imgs = []
    __dataset_label = []
    __dataset_imgs_val = []
    __dataset_label_val = []
    __num_images = 0
    __train_xs = []
    __train_ys = []
    __val_xs = []
    __val_ys = []
    __num_train_images = 0
    __num_val_images = 0
    __train_batch_pointer = 0
    __val_batch_pointer = 0
    __train_perc = 0
    __val_perc = 0
    __split_training = False

    # points to the end of the last batch
    __train_batch_pointer = 0
    __val_batch_pointer = 0

    def __init__(self, path='DrivingData.h5', path_val='', train_perc=0.8, val_perc=0.2, shuffle=True):
        self.__augment = AugmentBatch()
        self.__train_perc = train_perc
        self.__val_perc = val_perc
        print("Loading training data")
        # Handle HDF5/LMDB datasets (Load content to memory)
        self.handle_file_dataset(path,path_val,train_perc,val_perc,shuffle)

        # Allow split only if val_perc different than zero or
        if val_perc == 0 and path_val == '':
            self.__split_training = False

        # Get number of images
        self.__num_train_images = len(self.__train_xs)
        self.__num_val_images = len(self.__val_xs)
        print("Number training images: %d" % self.__num_train_images)
        print("Number validation images: %d" % self.__num_val_images)

    def shuffleData(self):
        '''Shuffle all data in memory'''
        # Shuffle data
        c = list(zip(self.__xs, self.__ys))
        random.shuffle(c)
        self.__xs, self.__ys = zip(*c)

        if self.__split_training == True:
            # Training set 80%
            self.__train_xs = self.__xs[:int(len(self.__xs) * self.__train_perc)]
            self.__train_ys = self.__ys[:int(len(self.__xs) * self.__train_perc)]

            # Validation set 20%
            self.__val_xs = self.__xs[-int(len(self.__xs) * self.__val_perc):]
            self.__val_ys = self.__ys[-int(len(self.__xs) * self.__val_perc):]
        else:
            # Training set 100%
            self.__train_xs = self.__xs
            self.__train_ys = self.__ys

    def LoadTrainBatch(self, batch_size, size_x=100, size_y=100, should_augment=False, do_resize=False):
        '''Load training batch, if batch_size=-1 load all dataset'''
        x_out = []
        y_out = []

        # If batch_size is -1 load the whole thing
        if batch_size == -1:
            batch_size = self.__num_train_images

        # Populate batch
        for i in range(0, batch_size):
            # Load image
            # image = scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images], mode="RGB")
            image = self.__train_xs[(self.__train_batch_pointer + i) % self.__num_train_images]
            # Crop top, resize to 66x200 and divide by 255.0
            if do_resize:
                image = scipy.misc.imresize(image, [size_x, size_y]).astype(image.dtype)
            image = image / 255.0
            x_out.append(image)
            label = self.__train_ys[(self.__train_batch_pointer + i) % self.__num_train_images]
            if do_resize:
                # Seems that the imresize change the color if input is grayscale and not uint
                # https://github.com/scipy/scipy/issues/4458
                label = scipy.misc.imresize(np.squeeze(label.astype(np.uint8)), [size_x, size_y]).astype(label.dtype)
                # Keep same shape ex: (100,100,1)
                label = np.expand_dims(label, axis=-1)
            y_out.append(label)
            self.__train_batch_pointer += batch_size

        # Augment dataset if needed
        if should_augment == True:
            # Augment training batch
            augmented_batch = self.__augment.augment(list(zip(x_out, y_out)))
            # Expand zip into list
            x_out, y_out = map(list, zip(*augmented_batch))


        return x_out, y_out

    def LoadValBatch(self, batch_size, size_x=100, size_y=100, do_resize=False):
        '''Load validation batch, if batch_size=-1 load all dataset'''
        x_out = []
        y_out = []

        # If batch_size is -1 load the whole thing
        if batch_size == -1:
            batch_size = self.__num_val_images

        for i in range(0, batch_size):
            # Load image
            # image = scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images], mode="RGB")
            image = self.__val_xs[(self.__val_batch_pointer + i) % self.__num_val_images]
            # Crop top, resize to 66x200 and divide by 255.0
            if do_resize:
                image = scipy.misc.imresize(image, [size_x, size_y]).astype(image.dtype)
            image = image / 255.0
            x_out.append(image)
            label = self.__train_ys[(self.__val_batch_pointer + i) % self.__num_val_images]
            if do_resize:
                # Seems that the imresize change the color if input is grayscale and not uint
                # https://github.com/scipy/scipy/issues/4458
                label = scipy.misc.imresize(np.squeeze(label.astype(np.uint8)), [size_x, size_y]).astype(label.dtype)
                # Keep same shape ex: (100,100,1)
                label = np.expand_dims(label, axis=-1).astype(label.dtype)
            y_out.append(label)
            self.__val_batch_pointer += batch_size
        return x_out, y_out

    def get_num_images(self):
        return self.__num_images


    def get_dataset_set(self):
        '''Get all training+validation set'''
        return list(self.__xs), list(self.__ys)

    def handle_file_dataset(self, path_train, path_val='', train_perc=0.8, val_perc=0.2, shuffle=True):
        '''Handle loading HDF5 and LMDB files'''

        # Check if validation-set exist if not partition from training
        has_validation = not path_val == ''

        print('LMDB file')
        env = lmdb.open(path_train, readonly=True)

        # Iterate file and load items on memory
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode('ascii')
                if 'label' in key_str:
                    # Get shape information from key name
                    info_key = key_str.split('_')
                    # Get image shape [2:None] means from index 2 to the end
                    shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                    label_data = np.frombuffer(value, dtype=np.uint8).reshape(shape_img).astype(np.float32)
                    label_data = np.expand_dims(label_data, axis=2)
                    self.__ys.append(label_data)
                else:
                    # Get shape information from key name
                    info_key = key_str.split('_')
                    # Get image shape [2:None] means from index 2 to the end
                    shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                    self.__xs.append(np.frombuffer(value, dtype=np.uint8).reshape(shape_img).astype(np.float32))

        self.__num_images = len(self.__xs)

        # Create a zip list with images and angles
        c = list(zip(self.__xs, self.__ys))

        # Shuffle data
        if shuffle:
            random.shuffle(c)

        # Split the items on c
        self.__xs, self.__ys = zip(*c)

        # Check if validation set is not given
        if not has_validation:
            print('Spliting training and validation')
            self.__split_training = True
            # Training set 80%
            self.__train_xs = self.__xs[:int(len(self.__xs) * train_perc)]
            self.__train_ys = self.__ys[:int(len(self.__xs) * train_perc)]

            # Validation set 20%
            self.__val_xs = self.__xs[-int(len(self.__xs) * val_perc):]
            self.__val_ys = self.__ys[-int(len(self.__xs) * val_perc):]
        else:
            print('Load validation dataset')
            self.__split_training = False
            # Read lmdb
            env = lmdb.open(path_val, readonly=True)

            # Training set 100%
            self.__train_xs = self.__xs
            self.__train_ys = self.__ys

            # Iterate file and load items on memory
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode('ascii')
                    if 'label' in key_str:
                        # Get shape information from key name
                        info_key = key_str.split('_')
                        # Get image shape [2:None] means from index 2 to the end
                        shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                        label_data = np.frombuffer(value, dtype=np.uint8).reshape(shape_img).astype(np.float32)
                        label_data = np.expand_dims(label_data, axis=2)
                        self.__val_ys.append(label_data)
                    else:
                        # Get shape information from key name
                        info_key = key_str.split('_')
                        # Get image shape [2:None] means from index 2 to the end
                        shape_img = tuple(map(lambda x: int(x), info_key[2:None]))
                        self.__val_xs.append(np.frombuffer(value, dtype=np.uint8).reshape(shape_img).astype(np.float32))