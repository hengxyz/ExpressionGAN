import tensorflow as tf
import os

#from exprgan import ExprGAN
from exprgan_novgg import ExprGAN

flags = tf.app.flags
flags.DEFINE_integer(name='epoch', default=50, help='number of epochs')
flags.DEFINE_integer(name='batch_size', default=64, help='number of batch size')
flags.DEFINE_integer(name='y_dim', default=6, help='label dimension')
flags.DEFINE_integer(name='rb_dim', default=5, help='expression intensity range')
flags.DEFINE_boolean(name='is_train', default=True, help='training mode')
flags.DEFINE_string(name='dataset', default='OULU', help='dataset name')
flags.DEFINE_string(name='save_dir', default='save', help='dir for saving training results')
flags.DEFINE_string(name='test_dir', default='test', help='dir for testing images')
flags.DEFINE_string(name='checkpoint_dir', default='None', help='dir for loading checkpoints')
flags.DEFINE_boolean(name='is_stage_one', default=True, help='is it the first stage?')
flags.DEFINE_float('vgg_coeff', 1.0, 'vgg coefficient')
flags.DEFINE_float('q_coeff', 1.0, 'regularizer network coefficient')
flags.DEFINE_float('fm_coeff', 0.0, 'feature matching coefficient')
flags.DEFINE_float('gpu_memory_fraction', 1.0, 'gpu occupation')
flags.DEFINE_string(name='vgg_face', default='/data/zming/models/GAN/ExprGAN/vgg-face.mat', help='dir for saving split pickle annotation file')
flags.DEFINE_string(name='split_file', default='../split/oulu_anno.pickle', help='dir for saving split pickle annotation file')
flags.DEFINE_string(name='testing_sample_dir', default='/data/zming/datasets/test/mtcnn_align_128/0', help='dir for testing samples')
FLAGS = flags.FLAGS


def main(_):
    import pprint
    pprint.pprint(FLAGS.__flags)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        model = ExprGAN(
            session,
            is_training=FLAGS.is_train,
            save_dir=FLAGS.save_dir,
            dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir,
            is_stage_one=FLAGS.is_stage_one,
            size_batch=FLAGS.batch_size,
            y_dim=FLAGS.y_dim,
            rb_dim=FLAGS.rb_dim,
            vgg_coeff=FLAGS.vgg_coeff,
            q_coeff=FLAGS.q_coeff,
            fm_coeff=FLAGS.fm_coeff,
            vgg_face = FLAGS.vgg_face,
            split_file = FLAGS.split_file
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
            )
        else:
            seed = 2018
            print '\n\tTesting Mode'
            model.custom_test(
                testing_samples_dir=os.path.join(FLAGS.testing_sample_dir, '*.png'), random_seed=seed
            )


if __name__ == '__main__':

    tf.app.run()

