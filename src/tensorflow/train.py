<<<<<<< HEAD
import fire
import models
import tensorflow as tf
import os
from handle_data import HandleData
import model_util

class TrainModel(object):
    def __init__(self, gpu=0, logdir='./logsSC', savedir='./saveSC',
                 input='/mnt/fs3/QA_analitics/Person_Re_Identification/git_repo/datasets/PPSS/PPSS_LMDB_train',
                 input_val='/mnt/fs3/QA_analitics/Person_Re_Identification/git_repo/datasets/PPSS/PPSS_LMDB_validate', mem_frac=1):

        # Set enviroment variable to set the GPU to use
        if gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            print('Set tensorflow on CPU')
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.__logdir = logdir
        self.__savedir = savedir
        self.__input = input
        self.__input_val = input_val
        self.__memfrac = mem_frac

    def train(self, mode='segnet_connected', epochs=600, learning_rate_init=0.0001, checkpoint='', batch_size=10, l2_reg=0.0001):
        # Avoid allocating the whole memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.__memfrac)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        # Regularization value
        L2NormConst = l2_reg

        print('Train segmentation model:', mode)

        # Build model
        if mode.lower() == 'segnet':
            segmentation_model = models.SegnetNoConnected()
        elif mode.lower() == 'segnet_connected':
            segmentation_model = models.SegnetConnected()
        elif mode.lower() == 'segnet_connected_gate':
            segmentation_model = models.SegnetConnectedGate()
        else:
            segmentation_model = models.FullyConvolutionalNetworks()

        # Get Placeholders
        model_in = segmentation_model.input
        model_out = segmentation_model.output
        labels_in = segmentation_model.label_in
        anotation_prediction = segmentation_model.anotation_prediction

        # Add input image on summary
        tf.summary.image("input_image", model_in, 2)
        tf.summary.image("ground_truth", tf.cast(model_util.label2show(labels_in), tf.uint8), max_outputs=2)
        # Expand dimension before asking a sumary
        anotation_prediction = model_util.label2show(anotation_prediction)
        tf.summary.image("pred_annotation", tf.cast(tf.expand_dims(anotation_prediction, dim=3), tf.uint8),
                         max_outputs=2)


        # Get all model "parameters" that are trainable
        train_vars = tf.trainable_variables()

        # Add loss
        # Segmentation problems often uses this "spatial" softmax (Basically we want to classify each pixel)
        with tf.name_scope("SPATIAL_SOFTMAX"):
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out,labels=tf.squeeze(labels_in, squeeze_dims=[3]),name="spatial_softmax"))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

        # Add model accuracy
        with tf.name_scope("Loss_Validation"):
            loss_val = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out, labels=tf.squeeze(labels_in, squeeze_dims=[3]), name="spatial_softmax")))

        # Solver configuration
        # Get ops to update moving_mean and moving_variance from batch_norm
        # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("Solver"):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learning_rate_init
            # decay every 10000 steps with a base of 0.96
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       20000, 0.96, staircase=True)

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
        tf.summary.scalar("loos_val", loss_val)
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # Configure where to save the logs for tensorboard
        summary_writer = tf.summary.FileWriter(self.__logdir, graph=tf.get_default_graph())

        data = HandleData(path=self.__input, path_val=self.__input_val)
        num_images_epoch = int(data.get_num_images() / batch_size)
        print('Num samples', data.get_num_images(), 'Iterations per epoch:', num_images_epoch, 'batch size:',batch_size)

        # For each epoch
        for epoch in range(epochs):
            for i in range(int(data.get_num_images() / batch_size)):
                # Get training batch
                xs_train, ys_train = data.LoadTrainBatch(batch_size, should_augment=True)

                # Send training batch to tensorflow graph (Dropout enabled)
                train_step.run(feed_dict={model_in: xs_train, labels_in: ys_train})

                # Display some information each x iterations
                if i % 100 == 0:
                    # Get validation batch
                    v_xs, v_ys = data.LoadValBatch(batch_size)
                    # Send validation batch to tensorflow graph (Dropout disabled)
                    loss_value = loss_val.eval(feed_dict={model_in: v_xs, labels_in: v_ys})
                    print("Epoch: %d, Step: %d, Loss(Val): %g" % (epoch, epoch * batch_size + i, loss_value))

                # write logs at every iteration
                summary = merged_summary_op.eval(feed_dict={model_in: xs_train, labels_in: ys_train})
                summary_writer.add_summary(summary, epoch * batch_size + i)
            if epoch%10==0:
                # Save checkpoint after each epoch
                if not os.path.exists(self.__savedir):
                    os.makedirs(self.__savedir)
                checkpoint_path = os.path.join(self.__savedir, "model")
                filename = saver.save(sess, checkpoint_path, global_step=epoch)
                print("Model saved in file: %s" % filename)

            # Shuffle data at each epoch end
            print("Shuffle data")
            data.shuffleData()

if __name__ == '__main__':
  #fire.Fire(TrainModel)
  newtrain=TrainModel()
  newtrain.train()
=======
import fire
import models
import tensorflow as tf
import os
from handle_data import HandleData

class TrainModel(object):
    def __init__(self, gpu=0, logdir='./logs', savedir='./save', input='SegData', input_val='', mem_frac=0.8):

        # Set enviroment variable to set the GPU to use
        if gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            print('Set tensorflow on CPU')
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.__logdir = logdir
        self.__savedir = savedir
        self.__input = input
        self.__input_val = input_val
        self.__memfrac = mem_frac

    def train(self, mode='fcn', epochs=600, learning_rate_init=0.001, checkpoint='', batch_size=50, l2_reg=0.0001, nclass=151, do_resize=False):
        # Avoid allocating the whole memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.__memfrac)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        # Regularization value
        L2NormConst = l2_reg

        print('Train segmentation model:', mode)

        # Build model
        if mode.lower() == 'segnet':
            segmentation_model = models.SegnetNoConnected(num_classes=nclass)
        elif mode.lower() == 'segnet_connected':
            segmentation_model = models.SegnetConnected(num_classes=nclass)
        elif mode.lower() == 'segnet_connected_gate':
            segmentation_model = models.SegnetConnectedGate(num_classes=nclass)
        elif mode.lower() == 'fe_segmentation':
            segmentation_model = models.CAE_AutoEncoderFE_MaxPool(num_classes=nclass)
        else:
            segmentation_model = models.FullyConvolutionalNetworks(num_classes=nclass)

        # Get Placeholders
        model_in = segmentation_model.input
        model_out = segmentation_model.output
        labels_in = segmentation_model.label_in
        anotation_prediction = segmentation_model.anotation_prediction

        # Add input image on summary
        tf.summary.image("input_image", model_in, 2)
        tf.summary.image("ground_truth", tf.cast(labels_in, tf.uint8), max_outputs=2)
        # Expand dimension before asking a sumary
        tf.summary.image("pred_annotation", tf.cast(tf.expand_dims(anotation_prediction, dim=3), tf.uint8), max_outputs=2)

        # Get all model "parameters" that are trainable
        train_vars = tf.trainable_variables()

        # Add loss
        # Segmentation problems often uses this "spatial" softmax (Basically we want to classify each pixel)
        with tf.name_scope("SPATIAL_SOFTMAX"):
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out,labels=tf.squeeze(labels_in, squeeze_dims=[3]),name="spatial_softmax"))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

        # Add model accuracy
        with tf.name_scope("Loss_Validation"):
            loss_val = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out, labels=tf.squeeze(labels_in, squeeze_dims=[3]), name="spatial_softmax")))

        # Solver configuration
        # Get ops to update moving_mean and moving_variance from batch_norm
        # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("Solver"):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learning_rate_init
            # decay every 10000 steps with a base of 0.96
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       30000, 0.1, staircase=True)

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
        tf.summary.scalar("loss_val", loss_val)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("global_step", global_step)
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # Configure where to save the logs for tensorboard
        summary_writer = tf.summary.FileWriter(self.__logdir, graph=tf.get_default_graph())

        data = HandleData(path=self.__input, path_val=self.__input_val)
        num_images_epoch = int(data.get_num_images() / batch_size)
        print('Num samples', data.get_num_images(), 'Iterations per epoch:', num_images_epoch, 'batch size:',batch_size)

        # For each epoch
        for epoch in range(epochs):
            for i in range(int(data.get_num_images() / batch_size)):
                # Get training batch
                xs_train, ys_train = data.LoadTrainBatch(batch_size, should_augment=True, do_resize=do_resize)

                # Send training batch to tensorflow graph (Dropout enabled)
                train_step.run(feed_dict={model_in: xs_train, labels_in: ys_train})

                # Display some information each x iterations
                if i % 100 == 0:
                    # Get validation batch
                    xs, ys = data.LoadValBatch(batch_size, do_resize=do_resize)
                    # Send validation batch to tensorflow graph (Dropout disabled)
                    loss_value = loss_val.eval(feed_dict={model_in: xs, labels_in: ys})
                    print("Epoch: %d, Step: %d, Loss(Val): %g" % (epoch, epoch * batch_size + i, loss_value))

                # write logs at every iteration
                summary = merged_summary_op.eval(feed_dict={model_in: xs_train, labels_in: ys_train})
                summary_writer.add_summary(summary, epoch * batch_size + i)

            # Save checkpoint after each epoch
            if not os.path.exists(self.__savedir):
                os.makedirs(self.__savedir)
            checkpoint_path = os.path.join(self.__savedir, "model")
            filename = saver.save(sess, checkpoint_path, global_step=epoch)
            print("Model saved in file: %s" % filename)

            # Shuffle data at each epoch end
            print("Shuffle data")
            data.shuffleData()

if __name__ == '__main__':
  fire.Fire(TrainModel)
>>>>>>> master
