import tensorflow as tf
from controller import Controller

flags = tf.app.flags
flags.DEFINE_integer(name='epoch', default=200, help='number of epochs')
flags.DEFINE_integer(name='batch_size', default=64, help='number of epochs')
flags.DEFINE_integer(name='y_dim', default=8, help='number of epochs')
flags.DEFINE_integer(name='rb_dim', default=5, help='number of epochs')
flags.DEFINE_boolean(name='is_train', default=True, help='training mode')
flags.DEFINE_string(name='dataset', default='RAF', help='dataset name')
flags.DEFINE_string(name='save_dir', default='save', help='dir for saving training results')
flags.DEFINE_string(name='test_dir', default='test', help='dir for testing images')
flags.DEFINE_string(name='checkpoint_dir', default='None', help='dir for loading checkpoints')
flags.DEFINE_boolean(name='is_vis', default=False, help='is it the first stage?')
flags.DEFINE_boolean(name='is_simple_q', default=False, help='is it the first stage?')
flags.DEFINE_integer(name='z_dim', default=50, help='number of epochs')
flags.DEFINE_string(name='split_file', default='../split/oulu_anno.pickle', help='dir for saving split pickle annotation file')
FLAGS = flags.FLAGS


def main(_):

    import pprint
    pprint.pprint(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = Controller(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.save_dir,
            dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir,
            size_batch=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
            rb_dim=FLAGS.rb_dim,
            is_simple_q=FLAGS.is_simple_q,
            num_z_channels=FLAGS.z_dim,
            epoch = FLAGS.epoch,
            split_file = FLAGS.split_file
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            model.train(
                num_epochs=FLAGS.epoch,
            )
        else:
            print '\n\tVisualization Mode'
            model.custom_visualize(testing_samples_dir=FLAGS.test_dir + '/*.jpeg')



if __name__ == '__main__':

    tf.app.run()


