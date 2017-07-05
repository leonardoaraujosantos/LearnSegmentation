import skimage
import skimage.filters
import skimage.color
import random
from random import randint
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

class AugmentBatch:

    def __init__(self):
        # Initialize seed
        #random.seed(42)
        # Create a list of functions that could be applied on the batch
        self.__list_func = [lambda img: self.convert_to_gray(img), lambda img: self.add_noise(img),
                            lambda img: self.add_gaussian(img), lambda img: self.convert_to_sepia(img),
                            lambda img: self.color_swap(img), lambda img: self.invert_color(img)]

    def augment(self, batch):
        # Roll the dice
        prob = random.random()

        # Half chance of nothing half do some augmentation
        if prob < 0.5:
            return batch
        else:
            # Do a copy of the batch
            new_batch = batch

            # Flip steering independent of other augmentations (Idea is to have more steering actions on training)
            batch_fliped = self.flip_horizontal(new_batch)

            # Do augmentations based on the lambda list __list_func
            idx = 0
            for (img, label) in batch_fliped:
                # Choose one operation to be applied on each image of the batch
                operation = randint(0, len(self.__list_func) - 1)
                # Choose the operation randomically
                img = self.__list_func[operation](img)
                batch_fliped[idx] = (img, label)
                idx += 1

            return batch_fliped

    def convert_to_gray(self, img):
        # Get each channel
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # To keep same number of channels add gray to each one.
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray
        return img

    def convert_to_sepia(self, img):
        # Get each channel
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # To keep same number of channels add gray to each one.
        img[:, :, 0] = 0.393 * r + 0.769 * g + 0.189 * b
        img[:, :, 1] = 0.349 * r + 0.686 * g + 0.168 * b
        img[:, :, 2] = 0.272 * r + 0.534 * g + 0.131 * b
        return img

    def add_noise(self, img):
        new_img = skimage.util.random_noise(img,var=0.001)
        return new_img

    def invert_color(self, img):
        new_img = skimage.util.invert(img)
        return new_img

    def add_gaussian(self, img):
        new_img = skimage.filters.gaussian(img,sigma=0.9, multichannel=True)
        return new_img

    def color_swap(self, img):
        new_img = img
        list_chanels = [0, 1, 2]
        random.shuffle(list_chanels)
        new_img[:, : ,0] = img[:, :, list_chanels[0]]
        new_img[:, :, 1] = img[:, :, list_chanels[1]]
        new_img[:, :, 2] = img[:, :, list_chanels[2]]
        return new_img

    # Flip both the image and the steering
    def flip_horizontal(self, batch):
        # Do a copy of the batch
        new_batch = batch
        idx = 0
        for (img, label) in new_batch:
            img = np.fliplr(img)
            label = np.fliplr(label)
            new_batch[idx] = (img, label)
            idx += 1
        return new_batch

    def display_batch(self, batch):
        for img, steering in batch:
            plt.imshow(img)
            plt.show()