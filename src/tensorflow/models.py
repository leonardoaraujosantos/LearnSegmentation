<<<<<<< HEAD
import tensorflow as tf
import model_util as util


class SegnetNoConnected(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder


        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 64, 128, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 128, 256, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 256, 512, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 512, 512, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (23, 23), 512, 512, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act, (3, 3), (25, 25), 512, 256, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act, (5, 5), (53, 53), 256, 128, 2, name="dconv3",
                                                   do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act, (5, 5), (110, 110), 128, 64, 2, name="dconv4",
                                                   do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (img_size, img_size), 64, num_classes, 2, name="dconv5",
                                                   do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class SegnetConnected(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder


        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 64, 128, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 128, 256, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 256, 512, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 512, 512, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (23, 23), 512, 512, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act+self.__conv4_act, (3, 3), (25, 25), 512, 256, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act+self.__conv3_act, (5, 5), (53, 53), 256, 128, 2, name="dconv3",
                                                   do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act+self.__conv2_act, (5, 5), (110, 110), 128, 64, 2, name="dconv4",
                                                   do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act+self.__conv1_act, (5, 5), (img_size, img_size), 64, num_classes, 2, name="dconv5",
                                                   do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class SegnetConnectedGate(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 64, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 64, 128, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 128, 256, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 256, 512, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 512, 512, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (23, 23), 512, 512, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act+util.gate_tensor(self.__conv4_act, name='gate4'),
            (3, 3), (25, 25), 512, 256, 1, name="dconv2",do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act+util.gate_tensor(self.__conv3_act, name='gate3'),
            (5, 5), (53, 53), 256, 128, 2, name="dconv3",do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act+util.gate_tensor(self.__conv2_act, name='gate2'),
            (5, 5), (110, 110), 128, 64, 2, name="dconv4",do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act+util.gate_tensor(self.__conv1_act, name='gate1'),
            (5, 5), (img_size, img_size), 64, num_classes, 2, name="dconv5",do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act

=======
import tensorflow as tf
import model_util as util


class SegnetNoConnected(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder
        self.__size_input_x = img_size
        self.__size_input_y = img_size


        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 32, 64, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 64, 64, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (23, 23), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act, (3, 3), (25, 25), 64, 64, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act, (5, 5), (53, 53), 64, 64, 2, name="dconv3",
                                                   do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act, (5, 5), (110, 110), 64, 32, 2, name="dconv4",
                                                   do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (img_size, img_size), 32, num_classes, 2, name="dconv5",
                                                   do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')


    @property
    def input_size(self):
        return self.__size_input_x, self.__size_input_y

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class SegnetConnected(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder
        self.__size_input_x = img_size
        self.__size_input_y = img_size


        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 32, 64, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 64, 64, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (23, 23), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act+self.__conv4_act, (3, 3), (25, 25), 64, 64, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act+self.__conv3_act, (5, 5), (53, 53), 64, 64, 2, name="dconv3",
                                                   do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act+self.__conv2_act, (5, 5), (110, 110), 64, 32, 2, name="dconv4",
                                                   do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act+self.__conv1_act, (5, 5), (img_size, img_size), 32, num_classes, 2, name="dconv5",
                                                   do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')


    @property
    def input_size(self):
        return self.__size_input_x, self.__size_input_y

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class SegnetConnectedGate(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder
        self.__size_input_x = img_size
        self.__size_input_y = img_size

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 32, 64, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 64, 64, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (23, 23), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act+util.gate_tensor(self.__conv4_act, name='gate4'),
            (3, 3), (25, 25), 64, 64, 1, name="dconv2",do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act+util.gate_tensor(self.__conv3_act, name='gate3'),
            (5, 5), (53, 53), 64, 64, 2, name="dconv3",do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act+util.gate_tensor(self.__conv2_act, name='gate2'),
            (5, 5), (110, 110), 64, 32, 2, name="dconv4",do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act+util.gate_tensor(self.__conv1_act, name='gate1'),
            (5, 5), (img_size, img_size), 32, num_classes, 2, name="dconv5",do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')


    @property
    def input_size(self):
        return self.__size_input_x, self.__size_input_y

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act


class AutoEncoderSegnet(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder
        self.__size_input_x = img_size
        self.__size_input_y = img_size

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 224x224x3 after CONV 5x5 P:0 S:2 H_out: 1 + (224-5)/2 = 110, W_out= 1 + (224-5)/2 = 110
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 32, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 110x110x24 after CONV 5x5 P:0 S:2 H_out: 1 + (110-5)/2 = 53, W_out= 1 + (110-5)/2 = 53
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 32, 64, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 53x53x36 after CONV 5x5 P:0 S:2 H_out: 1 + (53-5)/2 = 25, W_out= 1 + (53-5)/2 = 25
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 64, 64, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 25x25x48 after CONV 3x3 P:0 S:1 H_out: 1 + (25-3)/1 = 23, W_out= 1 + (25-3)/1 = 23
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 64, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 23x23x64 after CONV 3x3 P:0 S:1 H_out: 1 + (23-3)/1 = 21, W_out=  1 + (23-3)/1 = 21
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 32, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        # CONV6: Input 21x21x32 after CONV 3x3 P:0 S:1 H_out: 1 + (21-3)/1 = 19, W_out=  1 + (21-3)/1 = 19
        self.__conv6 = util.conv2d(self.__conv5_act, 3, 3, 32, 32, 1, "conv6", do_summary=False)
        self.__conv6_bn = util.batch_norm(self.__conv6, training_mode, name='bn_c6')
        self.__conv6_act = util.relu(self.__conv6_bn, do_summary=False)

        # CONV7: Input 19x19x64 after CONV 3x3 P:0 S:1 H_out: 1 + (19-3)/1 = 17, W_out=  1 + (19-3)/1 = 17
        self.__conv7 = util.conv2d(self.__conv6_act, 3, 3, 32, 32, 2, "conv7", do_summary=False)
        self.__conv7_bn = util.batch_norm(self.__conv7, training_mode, name='bn_c6')
        self.__conv7_act = util.relu(self.__conv7_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t7_out = util.conv2d_transpose(self.__conv7_act, (3, 3), (19, 19), 32, 32, 2, name="dconv1",
                                                   do_summary=False)
        self.__conv_t7_out_bn = util.batch_norm(self.__conv_t7_out, training_mode, name='bn_t_c7')
        self.__conv_t7_out_act = util.relu(self.__conv_t7_out_bn, do_summary=False)

        self.__conv_t6_out = util.conv2d_transpose(self.__conv_t7_out_act, (3, 3), (21, 21), 32, 32, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t6_out_bn = util.batch_norm(self.__conv_t6_out, training_mode, name='bn_t_c6')
        self.__conv_t6_out_act = util.relu(self.__conv_t6_out, do_summary=False)

        self.__conv_t5_out = util.conv2d_transpose(self.__conv_t6_out_act, (3, 3), (23, 23), 32, 64, 1, name="dconv3",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (25, 25), 64, 64, 1, name="dconv4",do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (5, 5), (53, 53), 64, 64, 2, name="dconv5",do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (5, 5), (110, 110), 64, 32, 2, name="dconv6",do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (5, 5), (img_size, img_size), 32, 3, 2, name="dconv7",do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = util.relu(self.__conv_t1_out_bn, do_summary = False)


    @property
    def input_size(self):
        return self.__size_input_x, self.__size_input_y

    @property
    def output(self):
        return self.__y


    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None

    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act



class CAE_AutoEncoderFE_MaxPool(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size=100, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__size_input_x = img_size
        self.__size_input_y = img_size
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 100x100x3 after CONV 3x3 P:0 S:1 H_out: 1 + (100-3)/1 = 98, W_out= 1 + (100-3)/1 = 98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 3, 3, 3, 16, 1, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 98x98x16 after CONV 3x3 P:0 S:2 H_out: 1 + (98-3)/2 = 48, W_out= 1 + (98-3)/2 = 48
        self.__conv2 = util.conv2d(self.__conv1_act, 3, 3, 16, 16, 1, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # Add Maxpool
        self.__conv2_mp_act = util.max_pool(self.__conv2_act, 2,2,2,name="maxpool1")

        # CONV3: Input 48x48x16 after CONV 3x3 P:0 S:1 H_out: 1 + (48-3)/1 = 46, W_out= 1 + (48-3)/1 = 46
        self.__conv3 = util.conv2d(self.__conv2_mp_act, 3, 3, 16, 32, 1, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 46x46x32 after CONV 3x3 P:0 S:2 H_out: 1 + (46-3)/2 = 22, W_out= 1 + (46-3)/2 = 22
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 32, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # Add Maxpool
        self.__conv4_mp_act = util.max_pool(self.__conv4_act, 2, 2, 2, name="maxpool2")

        # CONV5: Input 22x22x64 after CONV 3x3 P:0 S:1 H_out: 1 + (22-3)/1 = 20, W_out=  1 + (22-3)/1 = 20
        self.__conv5 = util.conv2d(self.__conv4_mp_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (22, 22), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(
            self.__conv_t5_out_act,
            (3, 3), (46, 46), 64, 32, 2, name="dconv2", do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(
            self.__conv_t4_out_act,
            (3, 3), (48, 48), 32, 16, 1, name="dconv3", do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(
            self.__conv_t3_out_act,
            (3, 3), (98, 98), 16, 16, 2, name="dconv4", do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(
            self.__conv_t2_out_act,
            (3, 3), (img_size, img_size), 16, num_classes, 1, name="dconv5", do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn


        with tf.name_scope('anotation_pred'):
            # Create the predicted annotation
            # Now we filter on the third dimension the the strogest pixels from a particular class
            # Returns the index with the largest value across axes of a tensor, on our case the index will be one of the
            # classes represented by the depth of our output (150 classes)
            self.__anotation = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")

            # Just some ops to print the output shape and the prediction
            #self.__anotation = tf.Print(self.__anotation, [tf.shape(self.__anotation)], name='PrintShapeAnnotation')
            #self.__y = tf.Print(self.__y, [tf.shape(self.__y)], name='PrintOutShape')



    @property
    def input_size(self):
        return self.__size_input_x, self.__size_input_y

    @property
    def output(self):
        return self.__y

    @property
    def anotation_prediction(self):
        return self.__anotation

    @property
    def input(self):
        if self.__use_placeholder:
            return self.__x
        else:
            return None

    @property
    def label_in(self):
        if self.__use_placeholder:
            return self.__label
        else:
            return None



    @property
    def conv5(self):
        return self.__conv5_act

    @property
    def conv4(self):
        return self.__conv4_act

    @property
    def conv3(self):
        return self.__conv3_act

    @property
    def conv2(self):
        return self.__conv2_act

    @property
    def conv1(self):
        return self.__conv1_act
>>>>>>> master
