"""
Implementation of example defense.
This defense loads inception v1 checkpoint and classifies all images using loaded checkpoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
import time
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
from mytools import load_path_label

tf.flags.DEFINE_string(
    'checkpoint_path', './defense_example/models/inception_v1/', 'Path to checkpoint for inception network.')
tf.app.flags.DEFINE_boolean(
    'restore', True, 'whether to resotre from checkpoint')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'Batch size to processing images')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'How many classes of the data set')
tf.app.flags.DEFINE_float(
    'learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer(
    'max_steps', 100000, 'The number of training times')
tf.app.flags.DEFINE_string(
    'pretrained_model_path', None, '')

FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with open(filepath, 'rb') as f:
            raw_image = imread(f, mode='RGB')
            image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
            image = (image / 255.0) * 2.0 - 1.0
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def main(_):
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    nb_classes = FLAGS.num_classes
    input_images = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, 3])
    input_labels = tf.placeholder(tf.float32, [None, nb_classes])

    learning_rate = FLAGS.learning_rate
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, end_points = inception.inception_v1(input_images, num_classes=110, is_training=True)

    variables_to_restore = slim.get_variables_to_restore()

    loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_labels)
    total_loss_op = tf.reduce_mean(loss_op)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)


    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(variables_to_restore)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(), ignore_missing_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            sess.run(init)
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)


        start = time.time()

        data_generator = load_path_label('labels.txt', batch_shape, onehot=True)
        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            _, total_loss = sess.run([train_op, total_loss_op], feed_dict={input_images: data[0], input_labels:data[1]})
            
            if np.isnan(total_loss):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size) /(time.time() - start)
                start = time.time()
                print('Step {:06d}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, total_loss, avg_time_per_step, avg_examples_per_second))
        # saver.save(sess, )

if __name__ == '__main__':
    tf.app.run()
