

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import *
from scipy.io import loadmat
import pickle
from time import gmtime, strftime
from datetime import datetime


class Controller(object):
    def __init__(self,
                 session,
                 size_image=128,
                 size_kernel=5,
                 size_batch=64,
                 num_encoder_channels=64,
                 num_z_channels=50,
                 y_dim=7,
                 num_gen_channels=1024,
                 enable_tile_label=True,
                 tile_ratio=1.0,
                 is_training=True,
                 save_dir='./save',
                 dataset_name='OULU',
                 is_flip=True,
                 checkpoint_dir='./checkpoint',
                 is_stage_one=True,
                 rb_dim=3,
                 is_simple_q=False,
                 epoch = 200,
                 split_file = '../split/oulu_anno.pickle'
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.label_value_range = (0, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.y_dim = y_dim
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.save_dir = os.path.join(save_dir, subdir)
        #self.save_dir = os.path.join(save_dir)
        self.dataset_name = dataset_name
        self.is_flip = is_flip
        self.checkpoint_dir = checkpoint_dir
        self.is_stage_one = is_stage_one
        self.rb_dim = rb_dim
        self.is_simple_q = is_simple_q
        self.epoch = epoch
        self.split_file = split_file

        print "\n\tLoading data"
        #self.data_X, self.data_y = self.load_anno('../split/' + self.dataset_name.lower() + '_anno.pickle')
        #self.data_X = [os.path.join("../data", self.dataset_name, x) for x in self.data_X]
        self.data_X, self.data_y = self.load_anno(self.split_file)
        self.data_X = [os.path.join(self.dataset_name, x) for x in self.data_X]
        imreadImg = imread(self.data_X[0])
        if len(imreadImg.shape) >= 3:
            self.num_input_channels = imread(self.data_X[0]).shape[-1]
        else:
            self.num_input_channels = 1
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )
        self.emo = tf.placeholder(
            tf.float32,
            [self.size_batch, self.y_dim],
            name='emotion_labels'
        )

        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        self.rb = tf.placeholder(
            tf.float32,
            [self.size_batch, self.y_dim * self.rb_dim],
            name='rb')
        print '\n\tBuilding graph ...'
        self.G = self.generator(
            z=self.z_prior,
            y=self.rb,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            reuse_variables=False
        )
        self.D_G, self.D_G_logits, _, self.D_cont_G = self.discriminator_img(
            image=self.G,
            y=self.emo,
            is_training=self.is_training
        )
        self.D_input, self.D_input_logits, _, _ = self.discriminator_img(
            image=self.input_image,
            y=self.emo,
            is_training=self.is_training,
            reuse_variables=True
        )
        self.D_img_loss_input = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(self.D_input_logits, tf.ones_like(self.D_input_logits))
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_input_logits), logits=self.D_input_logits)
        )
        self.D_img_loss_G = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.zeros_like(self.D_G_logits))
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_G_logits), logits=self.D_G_logits)
        )
        self.G_img_loss = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.ones_like(self.D_G_logits))
            tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.ones_like(self.D_G_logits), labels=self.D_G_logits)
        )

        self.D_cont_loss_fake_array = []
        for i in range(self.y_dim):
            self.label_per_class = self.rb[:, i * self.rb_dim:(i + 1) * self.rb_dim]
            loss = tf.reduce_mean(
                tf.square(self.D_cont_G[i] - self.label_per_class)
            )
            self.D_cont_loss_fake_array.append(loss)
        self.D_cont_loss_fake = tf.reduce_sum(self.D_cont_loss_fake_array)

        ## adding the L1 loss between the original image and the generated image
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss


        trainable_variables = tf.trainable_variables()
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        self.D_cont_loss_fake_summary = tf.summary.scalar('D_cont_loss_fake', self.D_cont_loss_fake)
        self.D_cont_G_summary = tf.summary.histogram('D_cont_G', self.D_cont_G)

        #self.saver = tf.train.Saver(max_to_keep=10)
        self.saver = tf.train.Saver(max_to_keep=3)

    def train(self,
              num_epochs=200,
              learning_rate=0.0002,
              beta1=0.5,
              decay_rate=1.0,
              enable_shuffle=True,
              use_trained_model=True,
              ):

        num_epochs = self.epoch
        # *********************************** optimizer **************************************************************
        size_data = len(self.data_X)

        #self.loss_EG = self.G_img_loss + self.D_cont_loss_fake
        ##adding the L1 loss for quickly converging
        self.loss_EG = self.G_img_loss + self.D_cont_loss_fake + self.EG_loss
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G + self.D_cont_loss_fake

        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        ##### !!!!!!!!!!!!! Very important update for tensorflow 1.6 !!!!!!!!!!!! ###############
        ##### Inside the tf.train.AdamOptimizer() will define the new global/sharing variables (by tf.get_variables),
        ##### however the reuse flag of tf.variable_scope is True, means that all the variables fllowing have been
        ##### already defined, so it will raise the error as 'the variables of G_fc/w/Adam is not existed.....'
        ##### It is strange that before the version tf1.6, it will not raise the error....
        ##### Thus, reuse flag in the tf.variable_scope() should be set to AUTO_REUSE which will automatically set the
        ##### reuse flag as False if the global variables have not been defined or True if they are aleady exists just
        ##### for using. Precisely, the reuse flag here should be false since  tf.train.AdamOptimizer() will defined
        ##### the new global variables corresponding to the weights and bias for Adam gradients optimisation intead of
        ##### usting the defined weights and bias. But we can not set the tf.variable_scope as reuse=False, this will
        ##### break the sharing of the global variables existed. Thus, the only solution is to set a variable scope as
        ##### reuse == tf.AUTO_REUSE, which means for the global variables defined whose reuse flag is True, other wise
        ##### for the new global variables to be defined whose reuse flag is False. I guess that in the older version
        ##### of the tensorflow, this set has been included in the function tf.train.AdamOptimizer(), that is reason
        ##### that we can run the program successfully in the older version.
        ##### !!!!!!!!!!!!! Very important update for tensorflow 1.6 !!!!!!!!!!!! ###############
        with tf.variable_scope('Train', reuse=tf.AUTO_REUSE): ### even we want to set reuse as False, it is impossible
                                                              ### the alternative solution is setting to AUTO_REUSE
                                                            ### which can chose the reuse as True or False automaically
            self.EG_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.G_variables
            )

            # opt = tf.train.AdamOptimizer(0.1, beta1=0.9, beta2=0.999, epsilon=0.1)
            # grads = opt.compute_gradients(loss=self.loss_EG, var_list=self.G_variables)
            # opt.apply_gradients(grads, global_step=self.EG_global_step)



            self.D_img_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Di,
                var_list=self.D_img_variables
        )

        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.D_img_loss_input_summary, self.D_img_loss_G_summary,
            self.G_img_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary,
            self.D_cont_loss_fake_summary, self.D_cont_G_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        sample_files = self.data_X[0:self.size_batch]
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
            is_flip=self.is_flip
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        save_batch_images(
                    batch_images=sample_images,
                    save_path=os.path.join(self.save_dir, 'sample.jpg'),
                    image_value_range=self.image_value_range,
                    size_frame=[self.y_dim, int(self.size_batch/self.y_dim)]

                )
        sample_label_emo = self.data_y[0:self.size_batch]
        sample_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim), dtype=np.float32)
        for i in range(self.size_batch):
            sample_label_rb[i] = self.y_to_rb_label(sample_label_emo[i])
        sample_z_prior = np.random.uniform(
            self.image_value_range[0],
            self.image_value_range[-1],
            [self.size_batch, self.num_z_channels]
        ).astype(np.float32)

        print '\n\tPreparing for training ...'
        tf.global_variables_initializer().run()

        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        num_batches = len(self.data_X) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                seed = 2017
                ## shuffle with the same random seed, thus the random order of data_x, data_y are the same; while if the
                ## seed is not reset, the order of the data_x and data_y will be two different random orders.
                np.random.seed(seed)
                np.random.shuffle(self.data_X)
                np.random.seed(seed)
                np.random.shuffle(self.data_y)
            for ind_batch in range(num_batches):
                start_time = time.time()
                batch_files = self.data_X[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                    is_flip=self.is_flip
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_label_emo = self.data_y[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch_label_rb = np.zeros(shape=(self.size_batch, self.rb_dim * self.y_dim),
                                           dtype=np.float32)
                for i in range(self.size_batch):
                    batch_label_rb[i] = self.y_to_rb_label(batch_label_emo[i])

                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                _, _, Gi_err, DiG_err, Di_err, DG_cont_err= self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_img_optimizer,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input,
                        self.D_cont_loss_fake
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.emo: batch_label_emo,
                        self.z_prior: batch_z_prior,
                        self.rb: batch_label_rb
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]" %
                    (epoch+1, num_epochs, ind_batch+1, num_batches))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))
                print("\tDG_cont=%.4f" % (DG_cont_err))

                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.emo: batch_label_emo,
                        self.z_prior: batch_z_prior,
                        self.rb: batch_label_rb
                    }
                )

                self.writer.add_summary(summary, self.EG_global_step.eval())

            name = '{:02d}.png'.format(epoch+1)
            ## Validation the trained generator by generating image by the sampled of the [z, rb]
            self.sample(sample_z_prior, sample_label_rb, name=name)

            if np.mod(epoch, 10) == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        self.writer.close()

    def generator(self, z, y, gender=None, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.y_dim)
        else:
            duplicate = 1
        z = concat_label(z, y, duplicate=duplicate)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1

        size_mini_map = int(self.size_image / 2 ** num_layers)

        name = 'G_fc'
        current = fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name
        )

        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)
        current = concat_label(current, y)

        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = tf.image.resize_nearest_neighbor(current, [size_mini_map * 2 ** (i + 1), size_mini_map * 2 ** (i + 1)])
            current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 1)), name=name)
            current = tf.nn.relu(current)
            current = concat_label(current, y)

        name = 'G_deconv' + str(i + 1)
        current = tf.image.resize_nearest_neighbor(current, [self.size_image, self.size_image])
        current = custom_conv2d(input_map=current, num_output_channels=int(self.num_gen_channels / 2 ** (i + 2)), name=name)
        current = tf.nn.relu(current)
        current = concat_label(current, y)

        name = 'G_deconv' + str(i + 2)
        current = custom_conv2d(input_map=current, num_output_channels=self.num_input_channels, name=name)

        return tf.nn.tanh(current)

    def discriminator_img(self, image, y, gender=None, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        current = concat_label(current, y)
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name
                )
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)
            current = concat_label(current, y)

        name = 'D_img_fc1'
        current = fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        name = 'D_img_fc1_bn'
        current = tf.contrib.layers.batch_norm(
            current,
            scale=False,
            is_training=is_training,
            scope=name,
            reuse=reuse_variables
        )
        current = lrelu(current)
        shared = concat_label(current, y)

        name = 'D_img_fc2'
        disc = fc(
            input_vector=shared,
            num_output_length=1,
            name=name
        )

        if self.is_simple_q:
            name = 'D_img_q_shared'
            q_shared = fc(
                input_vector=shared,
                num_output_length=128,
                name=name
            )
            name = 'D_img_q_shared_bn'
            q_shared = tf.contrib.layers.batch_norm(
                q_shared,
                scale=False,
                is_training=is_training,
                scope=name,
                reuse=reuse_variables
            )
            q_shared = lrelu(q_shared)

            cats = []
            for i in range(self.y_dim):
                name = 'D_img_q' + str(i)
                cat = fc(q_shared, self.rb_dim, name)
                cats.append(cat)
        else:
            name = 'D_img_q_shared'
            q_shared = fc(
                input_vector=shared,
                num_output_length=128,
                name=name
            )
            name = 'D_img_q_shared_bn'
            q_shared = tf.contrib.layers.batch_norm(
                q_shared,
                scale=False,
                is_training=is_training,
                scope=name,
                reuse=reuse_variables
            )
            q_shared = lrelu(q_shared)

            cats = []
            for i in range(self.y_dim):
                name = 'D_img_q_fc' + str(i)
                cat_fc = fc(q_shared, 64, name=name)

                name = 'D_img_q_fc_bn' + str(i)
                cat_fc = tf.contrib.layers.batch_norm(
                    cat_fc,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
                cat_fc = lrelu(cat_fc)

                name = 'D_img_q_cat' + str(i)
                cat = tf.nn.tanh(fc(cat_fc, self.rb_dim, name=name))
                cats.append(cat)

        return tf.nn.sigmoid(disc), disc, current, cats


    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        #checkpoint_dir = os.path.join(self.save_dir, subdir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

        rev_info_filename = os.path.join(self.save_dir, 'revision_info.txt')
        with open(rev_info_filename, "w") as text_file:
            text_file.write('image_value_range: (%f, %f)\n' % (self.image_value_range[0],self.image_value_range[1]))
            text_file.write('label_value_range: (%f, %f)\n' % (self.label_value_range[0], self.label_value_range[1]) )
            text_file.write('size_batch: %f\n' % self.size_image)
            text_file.write('size_kernel: %f\n' % self.size_kernel)
            text_file.write('size_batch: %f\n' % self.size_batch)
            text_file.write('num_encoder_channels: %f\n' % self.num_encoder_channels)
            text_file.write('num_z_channels: %f\n' % self.num_z_channels)
            text_file.write('num_gen_channels: %f\n' % self.num_gen_channels)
            text_file.write('enable_tile_label: %s\n' % self.enable_tile_label)
            text_file.write('tile_ratio: %f\n' % self.tile_ratio)
            text_file.write('is_training: %s\n' % self.is_training)
            text_file.write('save_dir: %s\n' % self.save_dir)
            text_file.write('dataset_name: %s\n' % self.dataset_name)
            text_file.write('is_flip: %s\n' % self.is_flip)
            text_file.write('checkpoint_dir: %s\n' % self.checkpoint_dir)
            text_file.write('is_stage_one: %s\n' % self.is_stage_one)
            text_file.write('rb_dim: %f\n' % self.rb_dim)
            text_file.write('is_simple_q: %s\n' % self.is_simple_q)


    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, 'checkpoint')
        print checkpoint_dir
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            print checkpoints_name
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, z, labels, gender=None, name=None):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        G = self.session.run(
            self.G,
            feed_dict={
                self.z_prior: z,
                self.rb: labels,
            }
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[self.y_dim, int(self.size_batch/self.y_dim)]
        )

    def visualize(self, z, gender=None, name=None):
        test_dir = os.path.join(self.save_dir, 'vis')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        image_frame_dim = 3
        batch_size = self.rb_dim * image_frame_dim
        query_z = np.tile(z, [batch_size, 1])
        for idx_emotion in range(self.y_dim):
            label = np.zeros(shape=(batch_size, self.rb_dim * self.y_dim))
            y_one_hot = np.zeros((batch_size, self.rb_dim))
            y = np.linspace(-1, 1, batch_size)
            idx_axis = 2
            y_one_hot[:, idx_axis] = y
            label[:, idx_emotion * self.rb_dim: (idx_emotion + 1) * self.rb_dim] = y_one_hot
            samples = self.session.run(
                self.G,
                feed_dict={
                    self.z_prior:query_z,
                    self.rb:label,
                })
            save_batch_images(batch_images=samples, save_path = './%s/%s_test_%d_%s.png' % (test_dir, name, idx_emotion, strftime("%Y%m%d%H%M%S", gmtime())),\
            image_value_range = self.image_value_range, size_frame=[1, self.rb_dim * image_frame_dim])


    def custom_visualize(self, testing_samples_dir):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")
        num_samples = 1
        z = np.random.uniform(self.image_value_range[0],
                              self.image_value_range[-1],
                              [num_samples, self.num_z_channels]
        ).astype(np.float32)
        self.visualize(z, name='vis')

    def load_anno(self, anno_file):
        print "\n\tLoading anno"
        anno = pickle.load(open(anno_file, 'rb'))
        anno = anno['train']
        X = anno.keys()
        y = anno.values()
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        y_vec = np.ones(shape=(len(y), self.y_dim), dtype=np.float32) * self.label_value_range[0]
        for i, label in enumerate(y):
            y_vec[i, label] = self.label_value_range[-1]
        return X, y_vec

    def y_to_rb_label(self, label):
        '''convert one hot label to rb type label'''
        number = np.argmax(label)
        one_hot = np.random.uniform(-1, 1, self.rb_dim)
        #rb = np.tile(-1*np.abs(one_hot), self.y_dim)
        rb = np.tile(np.zeros(self.rb_dim), self.y_dim)
        rb[number * self.rb_dim:(number + 1) * self.rb_dim] = np.abs(one_hot)
        return rb

