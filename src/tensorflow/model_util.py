import tensorflow as tf
import numpy as np
from math import sqrt, floor


# References:
# https://gist.github.com/kukuruza/03731dc494603ceab0c5
# https://github.com/tensorflow/tensorflow/issues/908
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
# http://stackoverflow.com/questions/35759220/how-to-visualize-learned-filters-on-tensorflow
# http://stackoverflow.com/questions/33783672/how-can-i-visualize-the-weightsvariables-in-cnn-in-tensorflow
# https://www.youtube.com/watch?v=5tW3y7lm7V0
# https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/debugging/embedding.ipynb
def put_kernels_on_grid(kernel, grid_Y, grid_X, pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    # Normalize weights
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y (Just to create a border on the grid)
    x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')
    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def conv2d(x, k_h, k_w, channels_in, channels_out, stride, name="conv", viewWeights=False, pad='VALID',
           do_summary=True):
    with tf.name_scope(name):
        # Define weights
        # Initialize weights with Xavier Initialization
        shape = [k_h, k_w, channels_in, channels_out]
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        w = tf.Variable(initializer(shape=shape), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")

        # Convolution
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)

        # Conv activation
        activation = conv + b

        if do_summary:
            # Add summaries for helping debug
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activation", activation)

        # Visualize weights if needed
        if viewWeights == True:
            # Get the power of 2 number closes to the floor(sqrt)
            sq_num = floor(sqrt(channels_out))
            # Calculate grid size
            first_dim = 8
            #remain_dim = channels_out-(first_dim**2)
            second_dim = channels_out // first_dim
            #print('ch_out:',channels_out,'first_dim:',first_dim,'second_dim:',second_dim)
            tf.summary.image("W_grid", put_kernels_on_grid(w, first_dim, second_dim), 1)

        return activation


# 2d Transposed convolution (Deconvolution)
def conv2d_transpose(x, kernel, out_size, channels_in, channels_out, stride, name="deconv", pad='VALID',
                     do_summary=True):
    with tf.name_scope(name):
        # Define weights (Notice that out/in channels are swapped on transposed conv)
        w = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], channels_out, channels_in], stddev=0.1),
                        name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")

        # Image output shape
        shape4D = [tf.shape(x)[0], out_size[0], out_size[1], channels_out]
        # Deconvolution (Transposed convolution)
        conv = tf.nn.conv2d_transpose(x, w, output_shape=shape4D, strides=[1, stride, stride, 1], padding=pad)

        # Conv activation
        activation = conv + b

        if do_summary:
            # Add summaries for helping debug
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activation", activation)

        return activation


def sigmoid(x, name="Sigmoid", do_summary=True):
    with tf.name_scope(name):
        activation = tf.sigmoid(x)

        if do_summary:
            # Add summaries for helping debug
            tf.summary.histogram("activation", activation)
    return activation


def relu(x, name="Relu", do_summary=True):
    with tf.name_scope(name):
        activation = tf.nn.relu(x)

        if do_summary:
            # Add summaries for helping debug
            tf.summary.histogram("activation", activation)
    return activation


# Batchnorm
# https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
# https://www.tensorflow.org/api_docs/python/tf/nn/fused_batch_norm
# https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
# http://ruishu.io/2016/12/27/batchnorm/
# https://stackoverflow.com/questions/40879967/how-to-use-batch-normalization-correctly-in-tensorflow
# https://stackoverflow.com/documentation/tensorflow/7909/using-batch-normalization#t=201611300538141458755
# When is_training is True the moving_mean and moving_variance need to be updated,
# by default the update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a
# dependency to the train_op
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#    train_op = optimizer.minimize(loss)
def batch_norm(x, is_training, name="batch_norm"):
    with tf.name_scope(name):
        # There is a flag called fused that switch implementation between
        # (tf.nn.fused_batch_norm and nn.batch_normalization), but on the documentation there is no difference between
        # them.
        activation = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, center=True, scale=True,
                                                  is_training=is_training, trainable=True,
                                                  updates_collections=None)  # fused=True, scope=name
    return activation


def max_pool(x, k_h, k_w, S, name="maxpool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, S, S, 1], padding='SAME')


def fc_layer(x, channels_in, channels_out, name="fc", do_summary=True):
    with tf.name_scope(name):
        # Initialize weights with Xavier Initialization
        shape = [channels_in, channels_out]
        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initializer(shape=shape), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")
        activation = tf.nn.relu(tf.matmul(x, w) + b)

        if do_summary:
            # Add summaries for helping debug
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activation", activation)
        return activation


def linear_layer(x, channels_in, channels_out, name="linear", do_summary=True):
    with tf.name_scope(name):
        # Initialize weights with Xavier Initialization
        shape = [channels_in, channels_out]
        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initializer(shape=shape), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="bias")
        activation = tf.matmul(x, w) + b

        if do_summary:
            # Add summaries for helping debug
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activation", activation)
        return activation


def bound_layer(val_in, bound_val, name="bound_scale"):
    with tf.name_scope(name):
        # Bound val_in between -1..1 and scale by multipling by bound_val
        activation = tf.multiply(tf.atan(val_in), bound_val)
        # Add summaries for helping debug
        tf.summary.histogram("val_in", val_in)
        tf.summary.histogram("activation", activation)
        return activation

# Let backprop modulate how much signal it wants ...
def gate_tensor(tensor_in, name="gate"):
    with tf.name_scope(name):
        mu = tf.Variable(tf.random_normal([1], stddev=0.35),name="modulate")
        tf.summary.scalar("plot_gate_"+name, tf.reshape(mu,[]))
        return tf.multiply(tf.sigmoid(mu), tensor_in)


# Define the huber loss (More resilient against outliers)
# https://en.wikipedia.org/wiki/Huber_loss
# http://web.stanford.edu/class/cs20si/lectures/slides_03.pdf
# https://github.com/devsisters/DQN-tensorflow/issues/16
# https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn/38363141
def huber_loss(labels, predidction, delta=1.0):
    with tf.name_scope('Huber_Loss'):
        # Calculate a residual difference (error)
        residual = tf.abs(labels - predidction)

        # Check if error is bigger than a delta
        condition = tf.less(residual, delta)

        # Absolute error
        small_res = 0.5 * tf.square(residual)

        # L2
        large_res = (delta * residual) - (0.5 * tf.square(delta))

        # Decide between L2 and absolute error
        return tf.where(condition, small_res, large_res)


# Do augmentation (flip) on the graph
# Some references:
# https://stackoverflow.com/questions/38920240/tensorflow-image-operations-for-batches
# https://stackoverflow.com/questions/39574999/tensorflow-tf-image-functions-on-an-image-batch
# https://www.tensorflow.org/api_docs/python/tf/map_fn
def augment_op(image_tensor, label_img_tensor):
    with tf.name_scope('augmentation'):
        def flip_image():
            # Map the flip_left_right to each image on your batch
            distorted_image = tf.map_fn(lambda img: tf.image.flip_left_right(img), image_tensor)
            distorted_label = tf.map_fn(lambda img: tf.image.flip_left_right(img), label_img_tensor)
            return distorted_image, distorted_label

        def no_flip_image():
            distorted_image = image_tensor
            distorted_label = label_img_tensor
            return distorted_image, distorted_label

        # Uniform variable in [0,1)
        flip_coin_flip = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        pred_flip = tf.less(flip_coin_flip, 0.5)
        # Randomically select doing color augmentation
        imgs, labels = tf.cond(pred_flip, flip_image, no_flip_image, name='if_horz_flip')
        return imgs, labels


# Caculate number of parameters
# This will not return the number in bytes just the "number" of parameters
def get_paremeter_size(train_variables):
    return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in train_variables])