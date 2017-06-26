import tensorflow as tf
import model_util as util


class FullyConvolutionalNetworks(object):
    def __init__(self, input=None, use_placeholder=True, training_mode=True, img_size = 224, num_classes=151):
        self.__x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name='IMAGE_IN')
        self.__label = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1], name='LABEL_IN')
        self.__dropout_prob = tf.placeholder(tf.float32, name='drop_prob')
        self.__use_placeholder = use_placeholder

        ##### ENCODER
        # Calculating the convolution output:
        # https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/convolutional_neural_networks.html
        # H_out = 1 + (H_in+(2*pad)-K)/S
        # W_out = 1 + (W_in+(2*pad)-K)/S
        # CONV1: Input 66x200x3 after CONV 5x5 P:0 S:2 H_out: 1 + (66-5)/2 = 31, W_out= 1 + (200-5)/2=98
        if use_placeholder:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(self.__x, 5, 5, 3, 24, 2, "conv1", viewWeights=True, do_summary=False)
        else:
            # CONV 1 (Mark that want visualization)
            self.__conv1 = util.conv2d(input, 5, 5, 3, 24, 2, "conv1", viewWeights=True, do_summary=False)

        self.__conv1_bn = util.batch_norm(self.__conv1, training_mode, name='bn_c1')
        self.__conv1_act = util.relu(self.__conv1_bn, do_summary=False)

        # CONV2: Input 31x98x24 after CONV 5x5 P:0 S:2 H_out: 1 + (31-5)/2 = 14, W_out= 1 + (200-5)/2=47
        self.__conv2 = util.conv2d(self.__conv1_act, 5, 5, 24, 36, 2, "conv2", do_summary=False)
        self.__conv2_bn = util.batch_norm(self.__conv2, training_mode, name='bn_c2')
        self.__conv2_act = util.relu(self.__conv2_bn, do_summary=False)

        # CONV3: Input 14x47x36 after CONV 5x5 P:0 S:2 H_out: 1 + (14-5)/2 = 5, W_out= 1 + (47-5)/2=22
        self.__conv3 = util.conv2d(self.__conv2_act, 5, 5, 36, 48, 2, "conv3", do_summary=False)
        self.__conv3_bn = util.batch_norm(self.__conv3, training_mode, name='bn_c3')
        self.__conv3_act = util.relu(self.__conv3_bn, do_summary=False)

        # CONV4: Input 5x22x48 after CONV 3x3 P:0 S:1 H_out: 1 + (5-3)/1 = 3, W_out= 1 + (22-3)/1=20
        self.__conv4 = util.conv2d(self.__conv3_act, 3, 3, 48, 64, 1, "conv4", do_summary=False)
        self.__conv4_bn = util.batch_norm(self.__conv4, training_mode, name='bn_c4')
        self.__conv4_act = util.relu(self.__conv4_bn, do_summary=False)

        # CONV5: Input 3x20x64 after CONV 3x3 P:0 S:1 H_out: 1 + (3-3)/1 = 1, W_out= 1 + (20-3)/1=18
        self.__conv5 = util.conv2d(self.__conv4_act, 3, 3, 64, 64, 1, "conv5", do_summary=False)
        self.__conv5_bn = util.batch_norm(self.__conv5, training_mode, name='bn_c5')
        self.__conv5_act = util.relu(self.__conv5_bn, do_summary=False)

        ##### DECODER (At this point we have 1x18x64
        # Kernel, output size, in_volume, out_volume, stride
        self.__conv_t5_out = util.conv2d_transpose(self.__conv5_act, (3, 3), (3, 20), 64, 64, 1, name="dconv1",
                                                   do_summary=False)
        self.__conv_t5_out_bn = util.batch_norm(self.__conv_t5_out, training_mode, name='bn_t_c5')
        self.__conv_t5_out_act = util.relu(self.__conv_t5_out_bn, do_summary=False)

        self.__conv_t4_out = util.conv2d_transpose(self.__conv_t5_out_act, (3, 3), (5, 22), 64, 48, 1, name="dconv2",
                                                   do_summary=False)
        self.__conv_t4_out_bn = util.batch_norm(self.__conv_t4_out, training_mode, name='bn_t_c4')
        self.__conv_t4_out_act = util.relu(self.__conv_t4_out_bn, do_summary=False)

        self.__conv_t3_out = util.conv2d_transpose(self.__conv_t4_out_act, (5, 5), (14, 47), 48, 36, 2, name="dconv3",
                                                   do_summary=False)
        self.__conv_t3_out_bn = util.batch_norm(self.__conv_t3_out, training_mode, name='bn_t_c3')
        self.__conv_t3_out_act = util.relu(self.__conv_t3_out_bn, do_summary=False)

        self.__conv_t2_out = util.conv2d_transpose(self.__conv_t3_out_act, (5, 5), (31, 98), 36, 24, 2, name="dconv4",
                                                   do_summary=False)
        self.__conv_t2_out_bn = util.batch_norm(self.__conv_t2_out, training_mode, name='bn_t_c2')
        self.__conv_t2_out_act = util.relu(self.__conv_t2_out_bn, do_summary=False)

        # Observe that the last deconv depth is the same as the number of classes
        self.__conv_t1_out = util.conv2d_transpose(self.__conv_t2_out_act, (5, 5), (66, 200), 24, num_classes, 2, name="dconv5",
                                                   do_summary=False)
        self.__conv_t1_out_bn = util.batch_norm(self.__conv_t1_out, training_mode, name='bn_t_c1')

        # Model output (It's not the segmentation yet...)
        self.__y = self.__conv_t1_out_bn

        # Create the predicted annotation
        # Now we filter on the third dimension the the strogest pixels from a particular class
        self.__anotation_pre = tf.argmax(self.__conv_t1_out_bn, dimension=3, name="prediction")
        # Expand dimension
        self.__anotation = tf.expand_dims(self.__anotation_pre, dim=3)

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
    def dropout_control(self):
        return self.__dropout_prob

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
