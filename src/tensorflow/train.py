import fire
import models
import tensorflow as tf
import os

class TrainModel(object):
    def __init__(self, gpu=0, logdir='./logs', savedir='./save'):

        # Set enviroment variable to set the GPU to use
        if gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            print('Set tensorflow on CPU')
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.__logdir = logdir
        self.__savedir = savedir

    def train(self, mode = 'fcn', epochs = 600, learning_rate_init = 0.001, checkpoint = ''):
        # Avoid allocating the whole memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        print('Train segmentation model:', mode)

        # Build model
        if mode.lower() == 'fcn':
            segmentation_model = models.FullyConvolutionalNetworks()
        elif mode.lower() == 'deconvnet':
            segmentation_model = models.FullyConvolutionalNetworks()
        elif mode.lower() == 'segnet':
            segmentation_model = models.FullyConvolutionalNetworks()
        else:
            segmentation_model = models.FullyConvolutionalNetworks()

        # Get Placeholders
        model_in = segmentation_model.input
        model_out = segmentation_model.output
        labels_in = segmentation_model.label_in
        anotation_prediction = segmentation_model.anotation_prediction
        model_drop = segmentation_model.dropout_control

        # Add input image on summary
        tf.summary.image("input_image", model_in, 2)
        tf.summary.image("ground_truth", tf.cast(labels_in, tf.uint8), max_outputs=2)
        tf.summary.image("pred_annotation", tf.cast(anotation_prediction, tf.uint8), max_outputs=2)

        # Get all model "parameters" that are trainable
        train_vars = tf.trainable_variables()

        # Add loss
        # Segmentation problems often uses this "spatial" softmax (Basically we want to classify each pixel)
        with tf.name_scope("SPATIAL_SOFTMAX"):
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out,labels=tf.squeeze(labels_in,squeeze_dims=[3]),name="spatial_softmax")))

        # Solver configuration
        # Get ops to update moving_mean and moving_variance from batch_norm
        # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("Solver"):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learning_rate_init
            # decay every 10000 steps with a base of 0.96
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       1000, 0.9, staircase=True)

            # Basically update the batch_norm moving averages before the training step
            # http://ruishu.io/2016/12/27/batchnorm/
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Initialize all random variables (Weights/Bias)
        sess.run(tf.global_variables_initializer())

        # Load checkpoint if needed
        if checkpoint != '':
            # Load tensorflow model
            print("Loading pre-trained model: %s" % checkpoint)
            # Create saver object to save/load training checkpoint
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, checkpoint)
        else:
            # Just create saver for saving checkpoints
            saver = tf.train.Saver(max_to_keep=None)

        # Monitor loss, learning_rate, global_step, etc...
        tf.summary.scalar("loss_train", loss)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("global_step", global_step)
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # Configure where to save the logs for tensorboard
        summary_writer = tf.summary.FileWriter(self.__logdir, graph=tf.get_default_graph())

        #data = HandleData(path=input_train_hdf5, path_val=input_val_hdf5)
        #num_images_epoch = int(data.get_num_images() / batch_size)
        #print('Num samples', data.get_num_images(), 'Iterations per epoch:', num_images_epoch, 'batch size:',batch_size)

if __name__ == '__main__':
  fire.Fire(TrainModel)